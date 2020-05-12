import copy
import shutil
import subprocess
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from pippin.base import ConfigBasedExecutable
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs, get_config, ensure_list, get_data_loc
from pippin.merge import Merger
from pippin.task import Task


class BiasCor(ConfigBasedExecutable):
    def __init__(self, name, output_dir, config, dependencies, options, global_config):
        base = get_data_loc(config.get("BASE", "surveys/des/bbc/bbc_5yr.input"))

        super().__init__(name, output_dir, config, base, "=", dependencies=dependencies)

        self.options = options
        self.config = config
        self.logging_file = os.path.join(self.output_dir, "output.log")
        self.global_config = get_config()

        self.merged_data = config.get("DATA")
        self.merged_iasim = config.get("SIMFILE_BIASCOR")
        self.merged_ccsim = config.get("SIMFILE_CCPRIOR")
        self.classifier = config.get("CLASSIFIER")
        self.make_all = config.get("MAKE_ALL_HUBBLE", True)
        self.use_recalibrated = config.get("USE_RECALIBRATED", False)

        self.bias_cor_fits = None
        self.cc_prior_fits = None
        self.data = None
        self.sim_names = [m.output["sim_name"] for m in self.merged_data]
        self.blind = np.any([m.output["blind"] for m in self.merged_data])
        self.output["blind"] = self.blind
        self.genversions = [m.output["genversion"] for m in self.merged_data]
        self.num_verions = [len(m.output["fitres_dirs"]) for m in self.merged_data]
        self.genversion = "_".join(self.sim_names) + "_" + self.classifier.name

        self.config_filename = f"BBC_{self.name}_{self.genversion}.input"  # Make sure this syncs with the tmp file name
        self.config_path = os.path.join(self.output_dir, self.config_filename)
        self.job_name = os.path.basename(self.config_path)
        self.fit_output_dir = os.path.join(self.output_dir, "output")
        self.done_file = os.path.join(self.fit_output_dir, f"SALT2mu_FITSCRIPTS/ALL.DONE")
        self.probability_column_name = self.classifier.output["prob_column_name"]
        self.output["prob_column_name"] = self.probability_column_name

        if self.use_recalibrated:
            new_name = self.probability_column_name.replace("PROB_", "CPROB_")
            self.logger.debug(f"Updating prob column name from {self.probability_column_name} to {new_name}. I hope it exists!")
            self.probability_column_name = new_name
        self.output["fit_output_dir"] = self.fit_output_dir

        num_dirs = self.num_verions[0]
        if num_dirs == 1:
            self.output["subdirs"] = ["SALT2mu_FITJOBS"]
        else:
            self.output["subdirs"] = [f"{i + 1:04d}" for i in range(num_dirs)]

        self.w_summary = os.path.join(self.fit_output_dir, "w_summary.csv")
        self.output["w_summary"] = self.w_summary
        self.output["m0dif_dirs"] = [os.path.join(self.fit_output_dir, s) for s in self.output["subdirs"]]
        self.output_plots = [
            os.path.join(m, f"{self.name}_{(str(int(os.path.basename(m))) + '_') if os.path.basename(m).isdigit() else ''}hubble.png")
            for m in self.output["m0dif_dirs"]
        ]
        if not self.make_all:
            self.output_plots = [self.output_plots[0]]
        self.logger.debug(f"Making {len(self.output_plots)} plots")

        self.muopts = self.config.get("MUOPTS", {})
        self.muopt_order = list(self.muopts.keys())
        self.output["muopts"] = self.muopt_order
        self.output["hubble_plot"] = self.output_plots

    def generate_w_summary(self):
        try:
            header = None
            rows = []
            for d in self.output["m0dif_dirs"]:
                wpath1 = os.path.join(d, "wfit_M0DIF_FITOPT000.COSPAR")
                wpath2 = os.path.join(d, "wfit_M0DIF_FITOPT000_MUOPT000.COSPAR")
                wpath = None
                if os.path.exists(wpath1):
                    wpath = wpath1
                elif os.path.exists(wpath2):
                    wpath = wpath2
                if wpath is not None:
                    with open(wpath) as f:
                        lines = f.read().splitlines()
                        header = ["VERSION"] + lines[0].split()[1:]
                        values = [os.path.basename(d)] + lines[1].split()
                        rows.append(values)
                else:
                    self.logger.warning(f"Cannot find file {wpath1} or {wpath2} when generating wfit summary")

            df = pd.DataFrame(rows, columns=header).apply(pd.to_numeric, errors="ignore")
            self.logger.info(f"wfit summary reporting mean w {df['w'].mean()}, see file at {self.w_summary}")
            df.to_csv(self.w_summary, index=False, float_format="%0.4f")
            return True
        except Exception as e:
            self.logger.exception(e, exc_info=True)
            return False

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug("Done file found, biascor task finishing")
            with open(self.done_file) as f:
                failed = False
                if "FAIL" in f.read():
                    self.logger.error(f"Done file reporting failure! Check log in {self.logging_file} and other logs")

                    log_files = [self.logging_file]

                    for dir in self.output["m0dif_dirs"]:
                        log_files += [f for f in os.listdir(dir) if f.upper().endswith(".LOG")]
                    self.scan_files_for_error(log_files, "FATAL ERROR ABORT", "QOSMaxSubmitJobPerUserLimit", "DUE TO TIME LIMIT")
                    return Task.FINISHED_FAILURE

                if not os.path.exists(self.w_summary):
                    wfiles = [os.path.join(d, f) for d in self.output["m0dif_dirs"] for f in os.listdir(d) if f.startswith("wfit_") and f.endswith(".LOG")]
                    m0files = [
                        os.path.join(d, f) for d in self.output["m0dif_dirs"] for f in os.listdir(d) if f.startswith("SALT2mu_F") and f.endswith(".M0DIF")
                    ]
                    for path in wfiles:
                        with open(path) as f2:
                            if "ERROR:" in f2.read():
                                self.logger.error(f"Error found in wfit file: {path}")
                                failed = True
                    for path in m0files:
                        with open(path) as f2:
                            for line in f2.readlines():
                                if "WARNING(SEVERE):" in line:
                                    self.logger.warning(f"File {path} reporting severe warning: {line}")
                                    self.logger.warning("You wont see this warning on a rerun, so look into it now!")

                    if self.generate_w_summary():
                        return Task.FINISHED_SUCCESS
                    else:
                        self.logger.error(
                            f"Hubble diagram failed to run. This might be a plotting issue, so not failing biascor, but please check this! {self.output_dir}"
                        )
                        return Task.FINISHED_SUCCESS  # Note this is probably a plotting issue, so don't rerun the biascor by returning FAILURE
                else:
                    self.logger.debug(f"Found {self.w_summary}, task finished successfully")
                    return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_name)

    def get_simfile_biascor(self, ia_sims):
        return ",".join([os.path.join(m.output["fitres_dirs"][0], m.output["fitopt_map"]["DEFAULT"]) for m in ia_sims])

    def get_simfile_ccprior(self, cc_sims):
        return None if cc_sims is None else ",".join([os.path.join(m.output["fitres_dirs"][0], m.output["fitopt_map"]["DEFAULT"]) for m in cc_sims])

    def write_input(self, force_refresh):
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

        self.output["fitopt_index"] = self.merged_data[0].output["fitopt_index"]

        self.set_property("simfile_biascor", self.bias_cor_fits)
        self.set_property("simfile_ccprior", self.cc_prior_fits)
        self.set_property("varname_pIa", self.probability_column_name)
        self.set_property("OUTDIR_OVERRIDE", self.fit_output_dir, assignment=": ")
        self.set_property("STRINGMATCH_IGNORE", " ".join(self.genversions), assignment=": ")

        for key, value in self.options.items():
            assignment = "="
            if key.upper().startswith("BATCH"):
                assignment = ": "
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
            self.set_property("WFITMUDIF_OPT", "-ompri 0.30 -dompri 0.01  -wmin -1.5 -wmax -0.5 -wsteps 201 -hsteps 121 -blind", assignment=": ")
        else:
            self.set_property("blindflag", 0, assignment="=")
            self.set_property("WFITMUDIF_OPT", "-ompri 0.30 -dompri 0.01  -wmin -1.5 -wmax -0.5 -wsteps 201 -hsteps 121", assignment=": ")

        bullshit_hack = ""
        for i, d in enumerate(self.data):
            if i > 0:
                bullshit_hack += "\nINPDIR+: "
            bullshit_hack += d
        self.set_property("INPDIR+", bullshit_hack, assignment=": ")

        # Set MUOPTS at top of file
        mu_str = ""
        muopt_prob_cols = {"DEFAULT": self.probability_column_name}
        for label in self.muopt_order:
            value = self.muopts[label]
            if mu_str != "":
                mu_str += "\nMUOPT: "
            mu_str += f"[{label}] "
            if value.get("SIMFILE_BIASCOR"):
                mu_str += f"simfile_biascor={self.get_simfile_biascor(value.get('SIMFILE_BIASCOR'))} "
            if value.get("SIMFILE_CCPRIOR"):
                mu_str += f"simfile_ccprior={self.get_simfile_ccprior(value.get('SIMFILE_CCPRIOR'))} "
            if value.get("CLASSIFIER"):
                cname = value.get("CLASSIFIER").output["prob_column_name"]
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
                        opt2 = self.probability_column_name
                    mu_str += f"CUTWIN {opt2} {opt_value}"
                else:
                    mu_str += f"{opt}={opt_value} "
            mu_str += "\n"
        if mu_str:
            self.set_property("MUOPT", mu_str, assignment=": ", section_end="#MUOPT_END")

        self.output["muopt_prob_cols"] = muopt_prob_cols
        final_output = "\n".join(self.base)

        new_hash = self.get_hash_from_string(final_output)
        old_hash = self.get_old_hash()

        if force_refresh or new_hash != old_hash:
            self.logger.debug("Regenerating results")

            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)

            with open(self.config_path, "w") as f:
                f.writelines(final_output)
            self.logger.info(f"Input file written to {self.config_path}")

            self.save_new_hash(new_hash)
            return True
        else:
            self.logger.debug("Hash check passed, not rerunning")
            return False

    def _run(self, force_refresh):
        if self.blind:
            self.logger.info("NOTE: This run is being BLINDED")
        regenerating = self.write_input(force_refresh)
        if regenerating:
            command = ["SALT2mu_fit.pl", self.config_filename, "NOPROMPT"]
            self.logger.debug(f"Will check for done file at {self.done_file}")
            self.logger.debug(f"Will output log at {self.logging_file}")
            self.logger.debug(f"Running command: {' '.join(command)}")
            with open(self.logging_file, "w") as f:
                subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
            chown_dir(self.output_dir)
        else:
            self.should_be_done()
            self.logger.info("Hash check passed, not rerunning")
        return True

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        merge_tasks = Task.get_task_of_type(prior_tasks, Merger)
        classifier_tasks = Task.get_task_of_type(prior_tasks, Classifier)
        tasks = []

        def _get_biascor_output_dir(base_output_dir, stage_number, biascor_name):
            return f"{base_output_dir}/{stage_number}_BIASCOR/{biascor_name}"

        for name in c.get("BIASCOR", []):
            config = c["BIASCOR"][name]
            options = config.get("OPTS", {})
            deps = []

            # Create dict but swap out the names for tasks
            # do this for key 0 and for muopts
            # modify config directly
            # create copy to start with to keep labels if needed
            config_copy = copy.deepcopy(config)

            def resolve_classifier(name):
                task = [c for c in classifier_tasks if c.name == name]
                if len(task) == 0:
                    Task.logger.info("CLASSIFIER {name} matched no classifiers. Checking prob column names instead.")
                    task = [c for c in classifier_tasks if c.get_prob_column_name() == name]
                    if len(task) == 0:
                        choices = [c.get_prob_column_name() for c in task]
                        message = f"Unable to resolve classifier {name} from list of classifiers {classifier_tasks} using either name or prob columns {choices}"
                        Task.fail_config(message)
                    if len(task) > 1:
                        Task.fail_config(f"Got {len(task)} prob column names? How is this even possible?")
                elif len(task) > 1:
                    choices = list(set([c.get_prob_column_name() for c in task]))
                    if len(choices) == 1:
                        task = [task[0]]
                    else:
                        Task.fail_config(f"Found multiple classifiers. Please instead specify a column name. Your choices: {choices}")
                return task[0]  # We only care about the prob column name

            def resolve_merged_fitres_files(name, classifier_name):
                task = [m for m in merge_tasks if classifier_name in m.output["classifier_names"] and m.output["lcfit_name"] == name]
                if len(task) == 0:
                    valid = [m.output["lcfit_name"] for m in merge_tasks]
                    message = f"Unable to resolve merge {name} from list of merge_tasks. There are valid options: {valid}"
                    Task.fail_config(message)
                elif len(task) > 1:
                    message = f"Resolved multiple merge tasks {task} for name {name}"
                    Task.fail_config(message)
                else:
                    return task[0]

            def resolve_conf(subdict, default=None):
                """ Resolve the sub-dictionary and keep track of all the dependencies """
                deps = []

                # If this is a muopt, allow access to the base config's resolution
                if default is None:
                    default = {}

                # Get the specific classifier
                classifier_name = subdict.get("CLASSIFIER")  # Specific classifier name
                if classifier_name is None and default is None:
                    Task.fail_config(f"You need to specify the name of a classifier under the CLASSIFIER key")
                classifier_task = None
                if classifier_name is not None:
                    classifier_task = resolve_classifier(classifier_name)
                classifier_dep = classifier_task or default.get("CLASSIFIER")  # For resolving merge tasks
                classifier_dep = classifier_dep.name
                if "CLASSIFIER" in subdict:
                    subdict["CLASSIFIER"] = classifier_task
                    deps.append(classifier_task)

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
            data_tasks = [resolve_merged_fitres_files(s, config["CLASSIFIER"].name) for s in data_names]
            deps += data_tasks
            config["DATA"] = data_tasks

            # Resolve every MUOPT
            muopts = config.get("MUOPTS", {})
            for label, mu_conf in muopts.items():
                deps += resolve_conf(mu_conf, default=config)

            task = BiasCor(name, _get_biascor_output_dir(base_output_dir, stage_number, name), config, deps, options, global_config)
            Task.logger.info(f"Creating aggregation task {name} with {task.num_jobs}")
            tasks.append(task)

        return tasks
