import copy
import shutil
import subprocess
import os
import pandas as pd
import numpy as np

from pippin.base import ConfigBasedExecutable
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs, get_config, ensure_list, get_data_loc, read_yaml, compress_dir
from pippin.merge import Merger
from pippin.task import Task


class BiasCor(ConfigBasedExecutable):
    def __init__(self, name, output_dir, config, dependencies, options, global_config):
        base = get_data_loc(config.get("BASE", "surveys/des/bbc/bbc_5yr.input"))
        self.base_file = base
        super().__init__(name, output_dir, config, base, "=", dependencies=dependencies)

        self.options = options
        self.logging_file = os.path.join(self.output_dir, "output.log")
        self.global_config = get_config()

        self.prob_cols = config["PROB_COLS"]

        self.merged_data = config.get("DATA")
        self.merged_iasim = config.get("SIMFILE_BIASCOR")
        self.merged_ccsim = config.get("SIMFILE_CCPRIOR")
        self.classifier = config.get("CLASSIFIER")
        if self.classifier is not None:
            self.config["CLASSIFIER"] = self.classifier.name
        self.make_all = config.get("MAKE_ALL_HUBBLE", True)
        self.use_recalibrated = config.get("USE_RECALIBRATED", False)
        self.consistent_sample = config.get("CONSISTENT_SAMPLE", True)
        self.bias_cor_fits = None
        self.cc_prior_fits = None
        self.data = None
        self.data_fitres = None
        self.sim_names = [m.output["sim_name"] for m in self.merged_data]
        self.blind = self.get_blind(config, options)
        self.logger.debug(f"Blinding set to {self.blind}")
        self.output["blind"] = self.blind
        self.genversions = [m.output["genversion"] for m in self.merged_data]
        self.num_verions = [len(m.output["fitres_dirs"]) for m in self.merged_data]
        self.output["fitopt_files"] = [m.output.get("fitopt_file") for m in self.merged_data]
        self.genversion = "_".join(self.sim_names) + ("" if self.classifier is None else "_" + self.classifier.name)

        self.config_filename = f"{self.name}.input"  # Make sure this syncs with the tmp file name
        self.config_path = os.path.join(self.output_dir, self.config_filename)
        self.kill_file = self.config_path.replace(".input", "_KILL.LOG")
        self.job_name = os.path.basename(self.config_path)
        self.fit_output_dir = os.path.join(self.output_dir, "output")
        self.merge_log = os.path.join(self.fit_output_dir, "MERGE.LOG")
        self.reject_list = os.path.join(self.output_dir, "reject.list")

        self.done_file = os.path.join(self.fit_output_dir, f"ALL.DONE")
        self.done_file_iteration = os.path.join(self.output_dir, "RESUBMITTED.DONE")
        self.run_iteration = 1 if os.path.exists(self.done_file_iteration) else 0
        self.probability_column_name = None
        if self.config.get("PROB_COLUMN_NAME") is not None:
            self.probability_column_name = self.config.get("PROB_COLUMN_NAME")
        elif self.classifier is not None:
            self.probability_column_name = self.prob_cols[self.classifier.name]
        self.output["prob_column_name"] = self.probability_column_name

        if self.use_recalibrated:
            new_name = self.probability_column_name.replace("PROB_", "CPROB_")
            self.logger.debug(f"Updating prob column name from {self.probability_column_name} to {new_name}. I hope it exists!")
            self.probability_column_name = new_name
        self.output["fit_output_dir"] = self.fit_output_dir

        self.output["NSPLITRAN"] = "NSPLITRAN" in [x.upper() for x in self.options.keys()]
        if self.output["NSPLITRAN"]:
            self.output["NSPLITRAN_VAL"] = {x.upper(): y for x, y in self.options.items()}["NSPLITRAN"]
        self.w_summary = os.path.join(self.fit_output_dir, "BBC_SUMMARY_wfit.FITRES")
        self.output["w_summary"] = self.w_summary

        self.set_m0dif_dirs()

        if not self.make_all:
            self.output_plots = [self.output_plots[0]]
        self.logger.debug(f"Making {len(self.output_plots)} plots")

        self.muopts = self.config.get("MUOPTS", {})
        self.muopt_order = list(self.muopts.keys())
        self.output["muopts"] = self.muopt_order
        self.output["hubble_plot"] = self.output_plots

        self.devel = self.options.get('devel', 0)

        self.logger.debug(f"Devel option: {self.devel}")
        self.do_iterate = False # Temp flag to stop iterating as BBC will reiterate natively
        self.logger.debug(f"Do iterate: {self.do_iterate}")
        self.logger.debug(f"SNANA_DIR: {os.environ['SNANA_DIR']}")

    def set_m0dif_dirs(self):

        versions = None
        # Check if the SUBMIT.INFO exists
        submit_info = os.path.join(self.fit_output_dir, "SUBMIT.INFO")
        if os.path.exists(submit_info):
            yml = read_yaml(submit_info)
            versions = yml.get("VERSION_OUT_LIST")

        if versions is not None:
            self.output["subdirs"] = versions
        else:
            num_dirs = self.num_verions[0]
            if self.output["NSPLITRAN"]:
                self.output["subdirs"] = [f"OUTPUT_BBCFIT-{i + 1:04d}" for i in range(self.output["NSPLITRAN_VAL"])]
            else:
                if num_dirs == 1:
                    self.output["subdirs"] = ["OUTPUT_BBCFIT"]
                else:
                    self.output["subdirs"] = [f"OUTPUT_BBCFIT-{i + 1:04d}" for i in range(num_dirs)]

        self.output["m0dif_dirs"] = [os.path.join(self.fit_output_dir, s) for s in self.output["subdirs"]]
        self.output_plots = [
            os.path.join(m, f"{self.name}_{(str(int(os.path.basename(m))) + '_') if os.path.basename(m).isdigit() else ''}hubble.png")
            for m in self.output["m0dif_dirs"]
        ]

    def get_blind(self, config, options):
        if "BLIND" in config:
            return config.get("BLIND")
        elif "blindflag" in options:
            return options.get("blindflag") != 0
        else:
            return bool(np.any([m.output["blind"] for m in self.merged_data]))

    def kill_and_fail(self):
        with open(self.kill_file, "w") as f:
            self.logger.info(f"Killing remaining jobs for {self.name}")
            command = ["submit_batch_jobs.sh", "--kill", os.path.basename(self.config_path)]
            subprocess.run([' '.join(command)], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir, shell=True)
        return Task.FINISHED_FAILURE

    def check_issues(self, kill=True):
        log_files = [self.logging_file]

        dirs = [self.output_dir, self.fit_output_dir, os.path.join(self.fit_output_dir, "SCRIPTS_BBCFIT")] + self.output["m0dif_dirs"]
        for dir in dirs:
            if os.path.exists(dir):
                log_files += [os.path.join(dir, f) for f in os.listdir(dir) if f.upper().endswith(".LOG")]
        self.scan_files_for_error(log_files, "FATAL ERROR ABORT", "QOSMaxSubmitJobPerUserLimit", "DUE TO TIME LIMIT")
        if kill:
            return self.kill_and_fail()
        else:
            return Task.FINISHED_FAILURE

    def submit_reject_phase(self):
        """ Merges the reject lists for each version, saves it to the output dir, modifies the input file, and resubmits if needed

        Returns: true if the job is resubmited, false otherwise
        """
        self.logger.info("Checking for rejected SNID after round 1 of BiasCor has finished")
        rejects = None
        for folder in self.output["m0dif_dirs"]:

            num_fitres_files = len([f for f in os.listdir(folder) if f.startswith("FITOPT") and ".FITRES" in f])
            if num_fitres_files < 2:
                self.logger.debug(f"M0DIF dir {folder} has only {num_fitres_files} FITRES file, so rejecting wont do anything. Not taking it into account.")
                continue
            path = os.path.join(folder, "BBC_REJECT_SUMMARY.LIST")
            df = pd.read_csv(path, delim_whitespace=True, comment="#")
            self.logger.debug(f"Folder {folder} has {df.shape[0]} rejected SNID")
            if rejects is None:
                rejects = df
            else:
                rejects = rejects.append(df)
        if rejects is None or not rejects.shape[0]:
            self.logger.info("No rejected SNIDs found, not rerunning, task finishing successfully")
            return Task.FINISHED_SUCCESS
        else:
            self.logger.info(f"Found {rejects.shape[0]} rejected SNIDs, will resubmit")
            self.logger.debug(f"Saving reject list to {self.reject_list}")
            rejects.to_csv(self.reject_list, sep=" ", index=False)

            # And now rerun
            if os.path.exists(self.done_file):
                os.remove(self.done_file)
            # Remove the output not just the done file
            tar_file = os.path.join(self.output_dir, "initial_bbc.tar.gz")
            moved = self.fit_output_dir + "_initial"
            self.logger.debug(f"Making tar of inital BBC run to {tar_file}")
            if os.path.exists(tar_file):
                os.remove(tar_file)

            shutil.move(self.fit_output_dir, moved)
            compress_dir(tar_file, moved)

            command = ["submit_batch_jobs.sh", os.path.basename(self.config_filename)]
            self.logger.debug(f"Running command: {' '.join(command)}")
            with open(self.logging_file, "w") as f:
                subprocess.run([' '.join(command)], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir, shell=True)
            self.logger.notice(f"RESUBMITTED: BiasCor {self.name} task")
            return 1

    def move_to_next_phase(self):
        if self.do_iterate and self.consistent_sample and self.run_iteration == 0:
            self.run_iteration += 1
            with open(self.done_file_iteration, "w") as f:
                pass
            return self.submit_reject_phase()
        else:
            self.logger.info(f"On run iteration {self.run_iteration}, finishing successfully")
            return Task.FINISHED_SUCCESS

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found for {self.name}, biascor task finishing")
            with open(self.done_file) as f:
                content = f.read().upper()
                if "FAIL" in content or "STOP" in content:
                    self.logger.error(f"Done file reporting failure! Check log in {self.logging_file} and other logs")
                    return self.check_issues()

                if not os.path.exists(self.w_summary):
                    self.logger.error(f"Generating w summary failed, please check this: {self.output_dir}")
                    return self.check_issues(kill=False)
                else:
                    self.logger.debug(f"Found {self.w_summary}, task finished successfully")
                    return self.move_to_next_phase()
        elif not os.path.exists(self.merge_log):
            self.logger.error("MERGE.LOG was not created, job died on submission")
            return self.check_issues()

        return self.check_for_job(squeue, self.job_name)

    def get_simfile_biascor(self, ia_sims):
        return None if ia_sims is None else ",".join([os.path.join(m.output["fitres_dirs"][0], "FITOPT000.FITRES.gz") for m in ia_sims])

    def get_simfile_ccprior(self, cc_sims):
        return None if cc_sims is None else ",".join([os.path.join(m.output["fitres_dirs"][0], "FITOPT000.FITRES.gz") for m in cc_sims])

    def get_fitopt_map(self, datas):
        fitopts = {}
        # Construct first map based off listed labels
        for data in datas:
            for label, file in data.output["fitopt_map"].items():
                if label == "DEFAULT":
                    continue
                if label not in fitopts:
                    fitopts[label] = {}
                fitopts[label][data.name] = file.split(".")[0]

        # If the label isnt present, then map it back to FITOPT000
        for label, d in fitopts.items():
            for data in datas:
                if fitopts[label].get(data.name) is None:
                    fitopts[label][data.name] = "FITOPT000"

        # Now for each of the labels we've found in all files, construct the output dict
        # Which is just FITOPT004: {DES_NAME: FITOPT004, LOWZ_NAME: FITOPT029}... etc
        index = 0
        result = {"SURVEY_LIST": " ".join([d.output["SURVEY"] for d in datas]), "FITOPT000": " ".join(["FITOPT000" for d in datas])}
        index_map = ["DEFAULT"]
        for label, mapping in fitopts.items():
            index += 1
            index_map.append(label)
            result[f"FITOPT{index:03d}"] = " ".join([mapping[d.name] for d in datas])

        return result, index_map

    def write_input(self):

        if self.merged_iasim is not None:
            for m in self.merged_iasim:
                if len(m.output["fitres_dirs"]) > 1:
                    self.logger.warning(f"Your IA sim {m} has multiple versions! Using 0 index from options {m.output['fitres_dirs']}")
        if self.merged_ccsim is not None:
            for m in self.merged_ccsim:
                if len(m.output["fitres_dirs"]) > 1:
                    self.logger.warning(f"Your CC sim {m} has multiple versions! Using 0 index from options {m.output['fitres_dirs']}")
        self.bias_cor_fits = self.get_simfile_biascor(self.merged_iasim)
        self.cc_prior_fits = self.get_simfile_ccprior(self.merged_ccsim)
        self.data = [m.output["lc_output_dir"] for m in self.merged_data]
        self.data_fitres = [m.output["fitres_file"] for m in self.merged_data]
        #print('MERGED DATA')
        #print(self.yaml)
        #print('------------')
        self.yaml["FITOPT_MAP"], fitopt_index = self.get_fitopt_map(self.merged_data)
        self.output["fitopt_index"] = fitopt_index

        if self.bias_cor_fits is not None:
            self.set_property("simfile_biascor", self.bias_cor_fits)
        if self.cc_prior_fits is not None:
            self.set_property("simfile_ccprior", self.cc_prior_fits)
        if self.probability_column_name is not None:
            self.set_property("varname_pIa", self.probability_column_name)

        self.yaml["CONFIG"]["OUTDIR"] = self.fit_output_dir

        yaml_keys = ["NSPLITRAN"]

        for key, value in self.options.items():
            assignment = "="
            if key.upper().startswith("BATCH") or key.upper() in yaml_keys:
                self.yaml["CONFIG"][key] = value
                continue
            if key.upper().startswith("CUTWIN"):
                assignment = " "
                split = key.split("_", 1)
                c = split[0]
                col = split[1]
                if col.upper() == "PROB_IA":
                    col = self.probability_column_name
                key = f"{c} {col}"
            self.set_property(key, value, assignment=assignment)

        if self.blind:
            self.set_property("blindflag", 2, assignment="=")
            w_string = self.yaml["CONFIG"].get("WFITMUDIF_OPT", "-ompri 0.311 -dompri 0.01  -wmin -1.5 -wmax -0.5 -wsteps 201 -hsteps 121") + " -blind"
            self.yaml["CONFIG"]["WFITMUDIF_OPT"] = w_string
        else:
            self.set_property("blindflag", 0, assignment="=")
            w_string = self.yaml["CONFIG"].get("WFITMUDIF_OPT", "-ompri 0.311 -dompri 0.01  -wmin -1.5 -wmax -0.5 -wsteps 201 -hsteps 121")
            self.yaml["CONFIG"]["WFITMUDIF_OPT"] = w_string

        # keys = [x.upper() for x in self.options.keys()]
        # if "NSPLITRAN" in keys:
        # No longer need to set STRINGMATCH_IGNORE for only one genversion?
        # NSPLITRAN updates means we dont need to dick around again
        # if "INPDIR+" in self.yaml["CONFIG"].keys():
        #     del self.yaml["CONFIG"]["INPDIR+"]
        # self.set_property("datafile", ",".join(self.data_fitres), assignment="=")
        # self.set_property("file", None, assignment="=")
        # else:
        # if len(self.data):
        #     self.yaml["CONFIG"]["STRINGMATCH_IGNORE"] = " ".join(self.genversions)

        self.yaml["CONFIG"]["INPDIR+"] = self.data

        # Set MUOPTS at top of file
        muopts = []
        muopt_scales = {}
        muopt_prob_cols = {"DEFAULT": self.probability_column_name}
        for label in self.muopt_order:
            value = self.muopts[label]
            muopt_scales[label] = value.get("SCALE", 1.0)
            mu_str = f"/{label}/ "
            if value.get("SIMFILE_BIASCOR"):
                mu_str += f"simfile_biascor={self.get_simfile_biascor(value.get('SIMFILE_BIASCOR'))} "
            if value.get("SIMFILE_CCPRIOR"):
                mu_str += f"simfile_ccprior={self.get_simfile_ccprior(value.get('SIMFILE_CCPRIOR'))} "
            if value.get("CLASSIFIER"):
                cname = self.prob_cols[value.get("CLASSIFIER").name]
                muopt_prob_cols[label] = cname
                mu_str += f"varname_pIa={cname} "
            else:
                muopt_prob_cols[label] = self.probability_column_name
            if value.get("FITOPT") is not None:
                mu_str += f"FITOPT={value.get('FITOPT')} "
            for opt, opt_value in value.get("OPTS", {}).items():
                self.logger.info(f"In MUOPT {label}, found OPTS flag for myopt with opt {opt} and value {opt_value}")
                if "CUTWIN_" in opt:
                    opt2 = opt.replace("CUTWIN_", "")
                    if opt2 == "PROB_IA":
                        opt2 = "varname_pIa"
                    mu_str += f"CUTWIN {opt2} {opt_value}"
                else:
                    mu_str += f"{opt}={opt_value} "
            muopts.append(mu_str)
        if muopts:
            self.yaml["CONFIG"]["MUOPT"] = muopts

        self.output["muopt_scales"] = muopt_scales
        self.output["muopt_prob_cols"] = muopt_prob_cols
        final_output = self.get_output_string()

        new_hash = self.get_hash_from_string(final_output)

        if self._check_regenerate(new_hash):
            self.logger.debug("Regenerating results")

            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            with open(self.reject_list, "w") as f:
                pass
            self.run_iteration = 0
            with open(self.config_path, "w") as f:
                f.writelines(final_output)
            self.logger.info(f"Input file written to {self.config_path}")

            self.save_new_hash(new_hash)
            return True
        else:
            self.logger.debug("Hash check passed, not rerunning")
            return False

    def _run(self):
        if self.blind:
            self.logger.info("NOTE: This run is being BLINDED")
        regenerating = self.write_input()
        if regenerating:
            command = ["submit_batch_jobs.sh", os.path.basename(self.config_filename)]
            self.logger.debug(f"Will check for done file at {self.done_file}")
            self.logger.debug(f"Will output log at {self.logging_file}")
            self.logger.debug(f"Running command: {' '.join(command)}")
            with open(self.logging_file, "w") as f:
                subprocess.run([' '.join(command)], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir, shell=True)
            chown_dir(self.output_dir)
            self.set_m0dif_dirs()
        else:
            self.should_be_done()
            self.logger.info("Hash check passed, not rerunning")
        return True

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        merge_tasks = Task.get_task_of_type(prior_tasks, Merger)
        prob_cols = {k: v for d in [t.output["classifier_merge"] for t in merge_tasks] for k, v in d.items()}
        classifier_tasks = Task.get_task_of_type(prior_tasks, Classifier)
        tasks = []

        def _get_biascor_output_dir(base_output_dir, stage_number, biascor_name):
            return f"{base_output_dir}/{stage_number}_BIASCOR/{biascor_name}"

        for name in c.get("BIASCOR", []):
            gname = name
            config = c["BIASCOR"][name]
            options = config.get("OPTS", {})
            deps = []

            # Create dict but swap out the names for tasks
            # do this for key 0 and for muopts
            # modify config directly
            # create copy to start with to keep labels if needed
            config_copy = copy.deepcopy(config)

            # Should return a single classifier task which maps to the desired prob column
            def resolve_classifiers(names):
                task = [c for c in classifier_tasks if c.name in names]
                if len(task) == 0:
                    if len(names) > 1:
                        Task.fail_config(f"CLASSIFIERS {names} do not match any classifiers. If these are prob column names, you must specify only one!")
                    Task.logger.info(f"CLASSIFIERS {names} matched no classifiers. Checking prob column names instead.")
                    task = [c for c in classifier_tasks if prob_cols[c.name] in names]
                    if len(task) == 0:
                        choices = [prob_cols[c.name] for c in task]
                        message = f"Unable to resolve classifiers {names} from list of classifiers {classifier_tasks} using either name or prob columns {choices}"
                        Task.fail_config(message)
                    else:
                        task = [task[0]]
                elif len(task) > 1:
                    choices = list(set([prob_cols[c.name] for c in task]))
                    if len(choices) == 1:
                        task = [task[0]]
                    else:
                        Task.fail_config(f"Found multiple classifiers. Please instead specify a column name. Your choices: {choices}")
                return task[0]  # We only care about the prob column name

            def resolve_merged_fitres_files(name, classifier_name):
                task = [m for m in merge_tasks if m.output["lcfit_name"] == name]
                if len(task) == 0:
                    valid = [m.output["lcfit_name"] for m in merge_tasks]
                    message = f"Unable to resolve merge {name} from list of merge_tasks. There are valid options: {valid}"
                    Task.fail_config(message)
                elif len(task) > 1:
                    message = f"Resolved multiple merge tasks {task} for name {name}"
                    Task.fail_config(message)
                else:
                    if classifier_name is not None and classifier_name not in task[0].output["classifier_names"]:
                        if prob_cols[classifier_name] not in [prob_cols[n] for n in task[0].output['classifier_names']]:
                            Task.logger.warning(
                                f"When constructing Biascor {gname}, merge input {name} does not have classifier {classifier_name}. "
                                f"If this is a spec confirmed sample, or an EXTERNAL task, all good, else check this."
                            )    
                    return task[0]

            # Ensure classifiers point to the same prob column
            def validate_classifiers(classifier_names):
                prob_col = []
                for name in classifier_names:
                    col = prob_cols.get(name)
                    if col is None:
                        # Check whether it is instead the prob_col name
                        if name in prob_cols.values():
                            prob_col.append(name)
                        else:
                            Task.fail_config(f"Classifier {name} has no prob column name in {prob_cols}. This should never happen!")
                    else:
                        prob_col.append(col)
                if len(set(prob_col)) > 1:
                    Task.fail_config(f"Classifiers {classifier_names} map to different probability columns: {prob_cols}, you may need to map them to the same name via MERGE_CLASSIFIERS in the AGGREGATION stage.")
                else:
                    Task.logger.debug(f"Classifiers {classifier_names} map to {prob_col[0]}")


            def resolve_conf(subdict, default=None):
                """ Resolve the sub-dictionary and keep track of all the dependencies """
                deps = []

                # If this is a muopt, allow access to the base config's resolution
                if default is None:
                    default = {}

                # Get the specific classifier
                classifier_names = subdict.get("CLASSIFIER")  # Specific classifier name
                if classifier_names is not None:
                    classifier_names = ensure_list(classifier_names)
                    validate_classifiers(classifier_names)
                #Task.logger.debug(f"XXX names: {classifier_names}")
                # Only if all classifiers point to the same prob_column should you continue
                classifier_task = None
                if classifier_names is not None:
                    classifier_task = resolve_classifiers(classifier_names)
                #Task.logger.debug(f"XXX tasks: {classifier_task}")
                classifier_dep = classifier_task or default.get("CLASSIFIER") # For resolving merge tasks
                if classifier_dep is not None:
                    classifier_dep = classifier_dep.name
                #Task.logger.debug(f"XXX deps: {classifier_dep}")
                if "CLASSIFIER" in subdict:
                    subdict["CLASSIFIER"] = classifier_task
                    if classifier_task is not None:
                        deps.append(classifier_task)
                #Task.logger.debug(f"XXX global deps: {deps}")

                # Get the Ia sims
                simfile_ia = subdict.get("SIMFILE_BIASCOR")
                if default is None and simfile_ia is None:
                    Task.fail_config(f"You must specify SIMFILE_BIASCOR for the default biascor. Supply a simulation name that has a merged output")
                if simfile_ia is not None:
                    simfile_ia = ensure_list(simfile_ia)
                    simfile_ia_tasks = [resolve_merged_fitres_files(s, classifier_dep) for s in simfile_ia]
                    deps += simfile_ia_tasks
                    subdict["SIMFILE_BIASCOR"] = simfile_ia_tasks

                # Resolve the cc sims
                simfile_cc = subdict.get("SIMFILE_CCPRIOR")
                if default is None and simfile_ia is None:
                    message = f"No SIMFILE_CCPRIOR specified. Hope you're doing a Ia only analysis"
                    Task.logger.warning(message)
                if simfile_cc is not None:
                    simfile_cc = ensure_list(simfile_cc)
                    simfile_cc_tasks = [resolve_merged_fitres_files(s, classifier_dep) for s in simfile_cc]
                    deps += simfile_cc_tasks
                    subdict["SIMFILE_CCPRIOR"] = simfile_cc_tasks

                return deps  # Changes to dict are by ref, will modify original

            deps += resolve_conf(config)
            # Resolve the data section
            data_names = config.get("DATA")
            if data_names is None:
                Task.fail_config("For BIASCOR tasks you need to specify an input DATA which is a mask for a merged task")
            data_names = ensure_list(data_names)
            class_task = config.get("CLASSIFIER")
            class_name = class_task.name if class_task is not None else None
            data_tasks = [resolve_merged_fitres_files(s, class_name) for s in data_names]
            deps += data_tasks
            config["DATA"] = data_tasks

            config["PROB_COLS"] = prob_cols

            # Resolve every MUOPT
            muopts = config.get("MUOPTS", {})
            for label, mu_conf in muopts.items():
                deps += resolve_conf(mu_conf, default=config)

            task = BiasCor(name, _get_biascor_output_dir(base_output_dir, stage_number, name), config, deps, options, global_config)
            Task.logger.info(f"Creating aggregation task {name} with {task.num_jobs}")
            tasks.append(task)

        return tasks
