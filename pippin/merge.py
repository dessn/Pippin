import shutil
import subprocess

from pippin.aggregator import Aggregator
from pippin.config import chown_dir, mkdirs
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

    def __init__(self, name, output_dir, config, dependencies, options):
        super().__init__(name, output_dir, config=config, dependencies=dependencies)
        self.options = options
        self.passed = False
        self.logfile = os.path.join(self.output_dir, "output.log")
        self.original_output = os.path.join(self.output_dir, "FITOPT000.FITRES.gz")
        self.lc_fit = self.get_lcfit_dep()
        self.agg = self.get_agg_dep()
        self.output["classifier_names"] = self.agg["classifier_names"]
        self.output["classifier_indexes"] = self.agg["classifier_indexes"]
        self.output["classifier_merge"] = self.agg["classifier_merge"]
        self.output["sim_name"] = self.lc_fit["sim_name"]
        self.output["lcfit_name"] = self.lc_fit["name"]
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

    def get_lcfit_dep(self):
        for d in self.dependencies:
            if isinstance(d, SNANALightCurveFit):
                return d.output
        msg = f"No dependency of a light curve fit task in {self.dependencies}"
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
        self.output["fitopt_map"] = self.lc_fit["fitopt_map"]
        self.output["fitopt_index"] = self.lc_fit["fitopt_index"]
        self.output["fitres_file"] = self.lc_fit["fitres_file"]
        self.output["SURVEY"] = self.lc_fit["SURVEY"]
        self.output["SURVEY_ID"] = self.lc_fit["SURVEY_ID"]

        fitres_files, symlink_files = [], []
        for index, (fitres_dir, outdir) in enumerate(
            zip(self.lc_fit["fitres_dirs"], self.fitres_outdirs)
        ):
            files = os.listdir(fitres_dir)
            fitres_files += [
                (fitres_dir, outdir, f, index, self.lc_fit["name"])
                for f in files
                if "FITRES" in f and not os.path.islink(os.path.join(fitres_dir, f))
            ]
            symlink_files += [
                (fitres_dir, outdir, f, index, self.lc_fit["name"])
                for f in files
                if "FITRES" in f and os.path.islink(os.path.join(fitres_dir, f))
            ]

        new_hash = self.get_hash_from_string(
            " ".join(
                [
                    a + b + c + f"{d}" + e
                    for a, b, c, d, e in (fitres_files + symlink_files)
                ]
            )
        )
        if self._check_regenerate(new_hash):
            shutil.rmtree(self.output_dir, ignore_errors=True)
            self.logger.debug("Regenerating, running combine_fitres")
            try:
                for fitres_dir in self.fitres_outdirs:
                    self.logger.debug(f"Creating directory {fitres_dir}")
                    mkdirs(fitres_dir)
                    for f in fitres_files:
                        if f[1] == fitres_dir:
                            self.add_to_fitres(
                                os.path.join(f[0], f[2]), f[1], f[4], index=f[3]
                            )
                    for s in symlink_files:
                        if s[1] == fitres_dir:
                            self.logger.debug(
                                f"Creating symlink for {os.path.join(s[1], s[2])} to {os.path.join(s[1], 'FITOPT000.FITRES.gz')}"
                            )
                            os.symlink(
                                os.path.join(s[1], "FITOPT000.FITRES.gz"),
                                os.path.join(s[1], s[2]),
                            )

                    self.logger.debug("Copying MERGE.LOG")
                    filenames = ["MERGE.LOG", "SUBMIT.INFO"]
                    for f in filenames:
                        original = os.path.join(self.lc_fit["lc_output_dir"], f)
                        moved = os.path.join(self.suboutput_dir, f)
                        if not os.path.exists(moved):
                            self.logger.debug(f"Copying file {f} into output directory")
                            shutil.copy(original, moved)

                    self.save_new_hash(new_hash)
                    with open(self.done_file, "w") as f:
                        f.write("SUCCESS\n")
            except Exception as e:
                self.logger.error("Error running merger!")
                self.logger.error(f"Check log at {self.logfile}")
                self.logger.exception(e, exc_info=True)
                return False
        else:
            self.should_be_done()
            self.logger.info("Hash check passed, not rerunning")
        return True

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        agg_tasks = Task.get_task_of_type(prior_tasks, Aggregator)
        lcfit_tasks = Task.get_task_of_type(prior_tasks, SNANALightCurveFit)
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
                    num_gen += 1

                    merge_name2 = f"{name}_{lcfit.name}"
                    task = Merger(
                        merge_name2,
                        _get_merge_output_dir(
                            base_output_dir, stage_number, name, lcfit.name
                        ),
                        config,
                        [lcfit, agg],
                        options,
                    )
                    Task.logger.info(
                        f"Creating merge task {merge_name2} for {lcfit.name} and {agg.name} with {task.num_jobs} jobs"
                    )
                    tasks.append(task)
            if num_gen == 0:
                Task.fail_config(
                    f"Merger {name} with mask {mask} matched no combination of aggregators and fits"
                )
        return tasks
