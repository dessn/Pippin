import copy
import inspect
import shutil
import subprocess
import os
import pandas as pd
from pippin.base import ConfigBasedExecutable
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs, get_config, ensure_list
from pippin.merge import Merger
from pippin.task import Task


class BiasCor(ConfigBasedExecutable):
    def __init__(self, name, output_dir, dependencies, options, config):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        base = config.get("BASE", "bbc.input")
        if "$" in base or base.startswith("/"):
            base = os.path.expandvars(base)
        else:
            base = os.path.join(self.data_dir, base)
        super().__init__(name, output_dir, base, "=", dependencies=dependencies)

        self.options = options
        self.config = config
        self.logging_file = os.path.join(self.output_dir, "output.log")
        self.global_config = get_config()

        self.merged_data = config.get("DATA")
        self.merged_iasim = config.get("SIMFILE_BIASCOR")
        self.merged_ccsim = config.get("SIMFILE_CCPRIOR")
        self.classifier = config.get("CLASSIFIER")

        self.bias_cor_fits = None
        self.cc_prior_fits = None
        self.data = None
        self.sim_names = [m.output["sim_name"] for m in self.merged_data]
        self.genversions = [m.output["genversion"] for m in self.merged_data]
        self.num_verions = [len(m.output["fitres_dirs"]) for m in self.merged_data]
        self.genversion = "_".join(self.sim_names) + "_" + self.classifier.name

        self.config_filename = f"{self.genversion}.input"  # Make sure this syncs with the tmp file name
        self.config_path = os.path.join(self.output_dir, self.config_filename)
        self.fit_output_dir = os.path.join(self.output_dir, "output")
        self.done_file = os.path.join(self.fit_output_dir, f"SALT2mu_FITSCRIPTS/ALL.DONE")
        self.probability_column_name = self.classifier.output["prob_column_name"]

        self.output["fit_output_dir"] = self.fit_output_dir

        # calculate genversion the hard way
        # Ricks 14Sep2019 update broke this
        # a_genversion = self.merged_data[0].output["genversion"]
        # for n in self.sim_names:
        #     a_genversion = a_genversion.replace(n, "")
        # while a_genversion.endswith("_"):
        #     a_genversion = a_genversion[:-1]

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

        self.output["muopts"] = self.config.get("MUOPTS", {}).keys()
        self.output["hubble_plot"] = self.output_plots

    def generate_w_summary(self):
        try:
            header = None
            rows = []
            for d in self.output["m0dif_dirs"]:
                wpath = os.path.join(d, "wfit_M0DIF_FITOPT000.COSPAR")
                if os.path.exists(wpath):
                    with open(wpath) as f:
                        lines = f.read().splitlines()
                        header = ["VERSION"] + lines[0].split()[1:]
                        values = [os.path.basename(d)] + lines[1].split()
                        rows.append(values)
                else:
                    self.logger.warning(f"Cannot find file {wpath} when generating wfit summary")

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
                    failed = True
                    self.logger.error(f"Done file reporting failure! Check log in {self.logging_file}")

                if not os.path.exists(self.w_summary):
                    wfiles = [os.path.join(d, f) for d in self.output["m0dif_dirs"] for f in os.listdir(d) if f.startswith("wfit_") and f.endswith(".LOG")]
                    for path in wfiles:
                        with open(path) as f2:
                            if "ERROR:" in f2.read():
                                self.logger.error(f"Error found in wfit file: {path}")
                                failed = True
                    plots_completed = self.make_hubble_plot()
                    if failed:
                        return Task.FINISHED_FAILURE
                    else:
                        self.generate_w_summary()
                        if plots_completed:
                            return Task.FINISHED_SUCCESS
                        else:
                            self.logger.error("Hubble diagram failed to run")
                            return Task.FINISHED_SUCCESS  # Note this is probably a plotting issue, so don't rerun the biascor by returning FAILURE
                else:
                    self.logger.debug(f"Found {self.w_summary}, task finished successfully")
                    return Task.FINISHED_SUCCESS
        if os.path.exists(self.logging_file):
            with open(self.logging_file) as f:
                output_error = False
                for line in f.read().splitlines():
                    if "ABORT ON FATAL ERROR" in line or "** ABORT **" in line:
                        self.logger.error(f"Output log showing abort: {self.logging_file}")
                        output_error = True
                    if output_error:
                        self.logger.error(line)
                if output_error:
                    return Task.FINISHED_FAILURE
        return 1

    def write_input(self, force_refresh):
        for m in self.merged_iasim:
            if len(m.output["fitres_dirs"]) > 1:
                self.logger.warning(f"Your IA sim {m} has multiple versions! Using 0 index from options {m.output['fitres_dirs']}")
        if self.merged_ccsim is not None:
            for m in self.merged_ccsim:
                if len(m.output["fitres_dirs"]) > 1:
                    self.logger.warning(f"Your CC sim {m} has multiple versions! Using 0 index from options {m.output['fitres_dirs']}")
        self.bias_cor_fits = ",".join([os.path.join(m.output["fitres_dirs"][0], m.output["fitopt_map"]["DEFAULT"]) for m in self.merged_iasim])
        self.cc_prior_fits = (
            None
            if self.merged_ccsim is None
            else ",".join([os.path.join(m.output["fitres_dirs"][0], m.output["fitopt_map"]["DEFAULT"]) for m in self.merged_ccsim])
        )
        self.data = [m.output["lc_output_dir"] for m in self.merged_data]

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

        bullshit_hack = ""
        for i, d in enumerate(self.data):
            if i > 0:
                bullshit_hack += "\nINPDIR+: "
            bullshit_hack += d
        self.set_property("INPDIR+", bullshit_hack, assignment=": ")

        # Set MUOPTS at top of file
        mu_str = ""
        for label, value in self.config.get("MUOPTS", {}).items():
            if mu_str != "":
                mu_str += "\nMUOPT: "
            mu_str += f"[{label}] "
            if value.get("SIMFILE_BIASCOR"):
                mu_str += f"simfile_biascor={','.join([v.output['fitres_file'] for v in value.get('SIMFILE_BIASCOR')])} "
            if value.get("SIMFILE_CCPRIOR"):
                mu_str += f"simfile_ccprior={','.join([v.output['fitres_file'] for v in value.get('SIMFILE_CCPRIOR')])} "
            if value.get("CLASSIFIER"):
                mu_str += f"varname_pIa={value.get('CLASSIFIER').output['prob_column_name']} "
            if value.get("FITOPT") is not None:
                mu_str += f"FITOPT={value.get('FITOPT')} "
            mu_str += "\n"
        if mu_str:
            self.set_property("MUOPT", mu_str, assignment=": ", section_end="#MUOPT_END")

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

    def make_hubble_plot(self):
        error = False
        if os.path.exists(self.output_plots[0]):
            self.logger.debug("Output plot exists")
        else:
            self.logger.info("Making output Hubble diagrams")
            for m, o in zip(self.output["m0dif_dirs"], self.output_plots):
                try:

                    fitres_file = os.path.join(m, "SALT2mu_FITOPT000_MUOPT000.FITRES")
                    m0dif_file = os.path.join(m, "SALT2mu_FITOPT000_MUOPT000.M0DIF")

                    from astropy.cosmology import FlatwCDM
                    import numpy as np
                    import matplotlib.pyplot as plt

                    df = pd.read_csv(fitres_file, comment="#", sep=r"\s+", skiprows=5)
                    dfm = pd.read_csv(m0dif_file, comment="#", sep=r"\s+", skiprows=10)
                    df.sort_values(by="zHD", inplace=True)
                    dfm.sort_values(by="z", inplace=True)
                    dfm = dfm[dfm["MUDIFERR"] < 10]

                    ol = 0.7
                    w = -1
                    alpha = 0
                    beta = 0
                    sigint = 0
                    gamma = r"$\gamma$ not fit"
                    num_sn_fit = df.shape[0]
                    contam_data, contam_true = "", ""
                    with open(m0dif_file) as f:
                        for line in f.read().splitlines():
                            if "Omega_DE(ref)" in line:
                                ol = float(line.strip().split()[-1])
                            if "w_DE(ref)" in line:
                                w = float(line.strip().split()[-1])
                    with open(fitres_file) as f:
                        for line in f.read().splitlines():
                            if "NSNFIT" in line:
                                v = int(line.split("=", 1)[1].strip())
                                num_sn_fit = v
                                num_sn = f"$N_{{SN}} = {v}$"
                            if "alpha0" in line and "=" in line and "+-" in line:
                                alpha = r"$\alpha = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
                            if "beta0" in line and "=" in line and "+-" in line:
                                beta = r"$\beta = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
                            if "sigint" in line and "iteration" in line:
                                sigint = r"$\sigma_{\rm int} = " + line.split()[3] + "$"
                            if "gamma" in line and "=" in line and "+-" in line:
                                gamma = r"$\gamma = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
                            if "CONTAM_TRUE" in line:
                                v = float(line.split("=", 1)[1].split("#")[0].strip())
                                n = v * num_sn_fit
                                contam_true = f"$R_{{CC, true}} = {v:0.4f} (\\approx {int(n)} SN)$"
                            if "CONTAM_DATA" in line:
                                v = float(line.split("=", 1)[1].split("#")[0].strip())
                                n = v * num_sn_fit
                                contam_data = f"$R_{{CC, data}} = {v:0.4f} (\\approx {int(n)} SN)$"

                    label = "\n".join([num_sn, alpha, beta, sigint, gamma, contam_true, contam_data])
                    zs = np.linspace(df["zHD"].min(), df["zHD"].max(), 500)
                    distmod = FlatwCDM(70, 1 - ol, w).distmod(zs).value

                    for log in [True, False]:

                        fig, axes = plt.subplots(figsize=(7, 5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0})

                        for resid, ax in enumerate(axes):

                            if log:
                                ax.set_xscale("log")

                            if resid:
                                sub = df["MUMODEL"]
                                sub2 = 0
                                sub3 = distmod
                                ax.set_ylabel(r"$\Delta \mu$")
                            else:
                                sub = 0
                                sub2 = -dfm["MUREF"]
                                sub3 = 0
                                ax.set_ylabel(r"$\mu$")
                                ax.annotate(label, (0.98, 0.02), xycoords="axes fraction", horizontalalignment="right", verticalalignment="bottom", fontsize=10)

                            ax.set_xlabel("$z$")

                            ax.errorbar(df["zHD"], df["MU"] - sub, yerr=df["MUERR"], fmt="none", elinewidth=0.5, c="#AAAAAA", alpha=0.5)
                            if df[self.probability_column_name].min() >= 1.0:
                                cc = df["IDSURVEY"]
                                vmax = None
                                color_prob = False
                                cmap = "rainbow"
                            else:
                                cc = df[self.probability_column_name]
                                vmax = 1.05
                                color_prob = True
                                cmap = "inferno"
                            h = ax.scatter(df["zHD"], df["MU"] - sub, c=cc, s=1, zorder=2, alpha=1, vmax=vmax, cmap=cmap)
                            ax.plot(zs, distmod - sub3, c="k", zorder=-1, lw=0.5, alpha=0.7)
                            ax.errorbar(dfm["z"], dfm["MUDIF"] - sub2, yerr=dfm["MUDIFERR"], fmt="o", elinewidth=1.0, c="k", ms=3)

                        if color_prob:
                            cbar = fig.colorbar(h, ax=axes, orientation="vertical", fraction=0.1, pad=0.01, aspect=40)
                            cbar.set_label("Prob Ia")
                        if log:
                            fp = o.replace(".png", "_log.png")
                        else:
                            fp = o
                        fig.savefig(fp, dpi=600, transparent=True, bbox_inches="tight")
                        plt.close(fig)
                except Exception as e:
                    self.logger.exception(e, exc_info=True)
                    error = True
        return not error

    def _run(self, force_refresh):
        regenerating = self.write_input(force_refresh)
        if regenerating:
            command = ["SALT2mu_fit.pl", self.config_filename, "NOPROMPT"]
            self.logger.debug(f"Will check for done file at {self.done_file}")
            self.logger.debug(f"Will output log at {self.logging_file}")
            self.logger.debug(f"Running command: {' '.join(command)}")
            with open(self.logging_file, "w") as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
            chown_dir(self.output_dir)
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
                    message = f"Unable to resolve merge merge {name} from list of merge_tasks {merge_tasks}"
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

            task = BiasCor(name, _get_biascor_output_dir(base_output_dir, stage_number, name), deps, options, config)
            Task.logger.info(f"Creating aggregation task {name} with {task.num_jobs}")
            tasks.append(task)

        return tasks
