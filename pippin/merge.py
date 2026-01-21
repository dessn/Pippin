from pippin.classifiers.classifier import Classifier
import shutil
from pathlib import Path
import subprocess

from pippin.aggregator import Aggregator
from pippin.config import chown_dir, mkdirs, get_data_loc, merge_dict, get_config
from pippin.dataprep import DataPrep
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from pippin.task import Task
import os


class Merger(Task):
    """Merge fitres files and aggregator output

    CONFIGURATION:
    ==============
    MERGE:
        label:
            MASK: partial match on all sim, fit and agg
            MASK_SIM: partial match on sim
            MASK_FIT: partial match on lcfit
            MASK_CLASS: partial match on classifier
            MASK_AGG: partial match on aggregation task

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        classifier_names: aggregators classifier names
        classifier_merge: Merge map of classifier task to prob column name
        sim_name: sim name being aggregated
        genversion: genverison of sim
        fitres_file: the location of the new FITOPT000.FITRES
        fitres_dir: the location of the directory in which the FITRES file lives
        fitopt_map: map from fitopt name (DEFAULT being nothing) to the FITOPTxxx.FITRES file
        lc_output_dir: path to the "output" folder created by split and fit
        lcfit_name: light curve fit name
        blind: bool - whether or not to blind cosmo results
    """

    def __new__(cls, name, output_dir, config, dependencies, options):
        # XXX DEPRECATION
        # If no BASE file is present, run legacy version of Merger
        # Avoid recursive nonsense by making sure the type of `cls` is Merger
        if cls == Merger and config.get("LEGACY"):
            # Have to import later because Merger must exist prior to importing MergerLegacy
            from pippin.merge_legacy import MergerLegacy

            cls = MergerLegacy
        return super().__new__(cls)

    def __init__(self, name, output_dir, config, dependencies, options):
        super().__init__(name, output_dir, config=config, dependencies=dependencies)

        base = get_data_loc(config.get("BASE", "submit_combine_fitres.input"))
        self.base_file = base

        self.options = options
        self.global_config = get_config()
        merge_input_base = os.path.basename(self.base_file)
        self.merge_output_file = self.output_dir + "/" + merge_input_base
        self.log_dir = f"{self.output_dir}/LOGS"
        self.total_summary = os.path.join(self.log_dir, "MERGE.LOG")
        self.done_file = f"{self.log_dir}/ALL.DONE"
        self.logging_file = self.merge_output_file.replace(".input", ".LOG")
        self.kill_file = self.merge_output_file.replace(".input", "_KILL.LOG")

        self.batch_file = self.options.get("BATCH_FILE")
        if self.batch_file is not None:
            self.batch_file = get_data_loc(self.batch_file)
        self.batch_replace = self.options.get(
            "BATCH_REPLACE", self.global_config.get("BATCH_REPLACE", {})
        )
        self.batch_replace["REPLACE_TEMPLATE"] = self.batch_replace.get(
            "REPLACE_TEMPLATE", self.global_config["SBATCH"]["cpu_location"]
        )

        self.passed = False
        self.logfile = os.path.join(self.output_dir, "output.log")
        self.original_output = os.path.join(self.output_dir, "FITOPT000.FITRES.gz")
        self.lc_fit = self.get_lcfit_dep()
        self.classifiers = self.get_class_deps()
        self.agg = self.get_agg_dep()
        self.output["classifier_names"] = self.agg["classifier_names"]
        self.output["classifier_indexes"] = self.agg["classifier_indexes"]
        self.output["classifier_merge"] = self.agg["classifier_merge"]
        self.output["sim_name"] = self.lc_fit["sim_name"]
        self.output["lcfit_name"] = self.lc_fit["name"]
        self.output["agg_name"] = self.agg.get("agg_name")
        self.output["genversion"] = self.lc_fit["genversion"]
        self.output["fitopt_file"] = self.lc_fit.get("fitopt_file")

        self.suboutput_dir = os.path.join(self.output_dir, "output")
        self.done_file = os.path.join(self.suboutput_dir, "ALL.DONE")

        self.fitres_outdirs = [
            os.path.join(self.suboutput_dir, os.path.basename(f))
            for f in self.lc_fit["fitres_dirs"]
        ]
        self.output["lc_output_dir"] = self.suboutput_dir
        self.output["fitres_dirs"] = self.fitres_outdirs
        self.output["genversion"] = self.lc_fit["genversion"]
        self.output["blind"] = self.lc_fit["blind"]

        print(f"XXX: merge\n{self.prepare_merge_input_lines()}")

    def prepare_merge_input_lines(self):
        merge_input_file = self.base_file

        # print(f"XXX: lcfit\n{__import__('pprint').pprint(self.lc_fit)}")
        # print(f"XXX: classifier\n{__import__('pprint').pprint(self.classifiers)}")

        with open(merge_input_file, "r") as i:
            config = i.read()
        header_dict = {
            "REPLACE_NAME": self.name,
            "REPLACE_WALLTIME": "1:00:00",
            "REPLACE_LOGFILE": "output.log",
            "REPLACE_MEM": "8GB",
            "REPLACE_TEMPLATE": "8GB",
            "APPEND": ["#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1"],
        }
        header_dict = merge_dict(header_dict, self.batch_replace)
        config = config.format(**header_dict).strip()

        task_template = """
- {REPLACE_TASK_NAME}
        INPUT_BASE: {REPLACE_INPUT_BASE}
        INPUT_APPEND: {REPLACE_INPUT_APPEND}
        OUTDIR_COMBINE: {REPLACE_OUTDIR_COMBINE}
        MIMIC_OUTDIR_SUBMIT_BATCH: {REPLACE_MIMIC_OUTDIR_SUBMIT_BATCH}
""".strip()

        task_name = self.lc_fit["genversion"]
        lcfit_fitres_dirs = self.lc_fit["fitres_dirs"]

        for fitres_dir in lcfit_fitres_dirs:
            for i, fitres in enumerate(Path(fitres_dir).iterdir()):
                outdir_combine = self.output_dir + "/output/" + fitres.parent.stem

                task_dict = {
                    "REPLACE_TASK_NAME": f"{task_name}-{str(i).rjust(3, '0')}",
                    "REPLACE_INPUT_BASE": str(fitres),
                    "REPLACE_INPUT_APPEND": str(
                        [
                            classifier["predictions_filename"]
                            for classifier in self.classifiers
                        ]
                    ),
                    "REPLACE_OUTDIR_COMBINE": outdir_combine,
                    "REPLACE_MIMIC_OUTDIR_SUBMIT_BATCH": f"{Path(fitres_dir).parent} {Path(outdir_combine).parent}",
                }
                task = task_template.format(**task_dict).strip()
                config += f"\n    {task}"

        return config

    def get_lcfit_dep(self):
        for d in self.dependencies:
            if isinstance(d, SNANALightCurveFit):
                return d.output
        msg = f"No dependency of a light curve fit task in {self.dependencies}"
        self.logger.error(msg)
        raise ValueError(msg)

    def get_class_deps(self):
        deps = []
        for d in self.dependencies:
            if isinstance(d, Classifier):
                deps.append(d.output)
        if len(deps) > 0:
            return deps
        msg = f"No dependency of a classifier task in {self.dependencies}"
        self.logger.error(msg)
        raise ValueError(msg)

    def get_agg_dep(self):
        for d in self.dependencies:
            if isinstance(d, Aggregator):
                return d.output
        msg = f"No dependency of an aggregator task in {self.dependencies}"
        self.logger.error(msg)
        raise ValueError(msg)

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(
                f"Merger finished, see combined fitres at {self.suboutput_dir}"
            )
            return Task.FINISHED_SUCCESS
        else:
            output_error = False
            if os.path.exists(self.logfile):
                with open(self.logfile, "r") as f:
                    for line in f.read().splitlines():
                        if "ERROR" in line or "ABORT" in line:
                            self.logger.error(
                                f"Fatal error in combine_fitres. See {self.logfile} for details."
                            )
                            output_error = True
                        if output_error:
                            self.logger.info(f"Excerpt: {line}")
                if output_error:
                    self.logger.debug("Removing hash on failure")
                    os.remove(self.hash_file)
                    chown_dir(self.output_dir)
            else:
                self.logger.error(
                    "Combine task failed with no output log. Please debug"
                )
            return Task.FINISHED_FAILURE

    def add_to_fitres(self, fitres_file, outdir, lcfit, index=0):
        if not self.agg["lcfit_names"]:
            lcfit_index = 0
        else:
            lcfit_index = self.agg["lcfit_names"].index(lcfit)

        if not self.agg["empty_agg"]:
            command = [
                "combine_fitres.exe",
                fitres_file,
                self.agg["merge_key_filename"][index][lcfit_index],
                "--outfile_text",
                os.path.basename(fitres_file),
                "T",
                "-nullval_float",
                "0",
            ]
            try:
                self.logger.debug(f"Executing command {' '.join(command)}")
                with open(self.logfile, "w+") as f:
                    subprocess.run(
                        command,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=outdir,
                        check=True,
                    )

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error invoking command {command}")
                raise e
        else:
            self.logger.info(
                "Empty aggregation result found, not invoking combine_fitres.exe"
            )
            self.logger.debug(f"Copying file {fitres_file} to {outdir}")
            shutil.copy(fitres_file, outdir)

    def _run(self):
        # TODO(@rkessler) Check this all works as expected
        # === START ===
        failed = False
        if Path(self.done_file).exists():
            self.logger.debug(f"Found done file at {self.done_file}")
            with open(self.done_file) as f:
                if "SUCCESS" not in f.read().upper():
                    failed = True
        # prepare merge input lines needed to create hash,
        # but don't create merge input file yet.
        merge_input_lines = self.prepare_merge_input_lines()
        str_config = " ".join(merge_input_lines)
        new_hash = self.get_hash_from_string(str_config)
        if self._check_regenerate(new_hash) or failed:
            self.logger.debug("Regenerating merger")
        else:
            self.should_be_done()
            self.logger.info("Hash check passed, not rerunning")
            return True

        shutil.rmtree(self.output_dir, ignore_errors=True)
        mkdirs(self.output_dir)

        # write merge output file
        with open(self.merge_output_file, "wt") as i:
            for line in merge_input_lines:
                i.write(f"{line}\n")

        self.save_new_hash(new_hash)

        with open(self.logging_file, "w") as f:
            subprocess.run(
                ["submit_batch_jobs.sh", os.path.basename(self.merge_output_file)],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=self.output_dir,
            )

        # === END ===
        self.output["fitopt_map"] = self.lc_fit["fitopt_map"]
        self.output["fitopt_index"] = self.lc_fit["fitopt_index"]
        self.output["fitres_file"] = self.lc_fit["fitres_file"]
        self.output["SURVEY"] = self.lc_fit["SURVEY"]
        self.output["SURVEY_ID"] = self.lc_fit["SURVEY_ID"]

        return True

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        agg_tasks = Task.get_task_of_type(prior_tasks, Aggregator)
        lcfit_tasks = Task.get_task_of_type(prior_tasks, SNANALightCurveFit)
        classify_tasks = Task.get_task_of_type(prior_tasks, Classifier)
        tasks = []

        def _get_merge_output_dir(
            base_output_dir, stage_number, merge_name, lcfit_name
        ):
            return f"{base_output_dir}/{stage_number}_MERGE/{merge_name}_{lcfit_name}"

        for name in c.get("MERGE", []):
            num_gen = 0
            config = c["MERGE"].get(name, {})
            if config is None:
                config = {}
            options = config.get("OPTS", {})
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_lc = config.get("MASK_FIT", "")
            mask_class = config.get("MASK_CLASS", "")
            mask_agg = config.get("MASK_AGG", "")

            for lcfit in lcfit_tasks:
                if mask and mask not in lcfit.name:
                    continue
                if mask_lc and mask_lc not in lcfit.name:
                    continue
                sim = lcfit.get_dep(SNANASimulation, DataPrep)
                if mask and mask not in sim.name:
                    continue
                if mask_sim and mask_sim not in sim.name:
                    continue

                for agg in agg_tasks:
                    if mask_agg and mask_agg not in agg.name:
                        continue
                    if mask and mask not in agg.name:
                        continue

                    # Check if the sim is the same for both
                    if sim != agg.get_underlying_sim_task():
                        continue

                    classifiers = []
                    for classify in classify_tasks:
                        if mask_class and mask_class not in classify.name:
                            continue
                        if mask and mask not in classify.name:
                            continue

                        # Check if the sim is the same for both
                        if (
                            classify.get_requirements(classify.options)[0]
                            and sim not in classify.get_simulation_dependency()
                        ):
                            continue
                        # Check if the lcfit is the same for both
                        if classify.get_requirements(classify.options)[
                            1
                        ] and lcfit not in classify.get_fit_dependency(output=False):
                            continue
                        classifiers.append(classify)
                    num_gen += 1

                    merge_name2 = f"{name}_{lcfit.name}"
                    task = Merger(
                        merge_name2,
                        _get_merge_output_dir(
                            base_output_dir, stage_number, name, lcfit.name
                        ),
                        config,
                        [lcfit, agg, *classifiers],
                        options,
                    )
                    Task.logger.info(
                        f"Creating merge task {merge_name2} for {lcfit.name}, {classify.name}, and {agg.name} with {task.num_jobs} jobs"
                    )
                    tasks.append(task)
            if num_gen == 0:
                Task.fail_config(
                    f"Merger {name} with mask {mask} matched no combination of aggregators and fits"
                )
        return tasks
