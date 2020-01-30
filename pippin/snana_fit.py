import inspect
import os
import shutil
import subprocess
import re
import pandas as pd
import numpy as np

from pippin.base import ConfigBasedExecutable
from pippin.config import mkdirs, get_data_loc, chown_dir
from pippin.dataprep import DataPrep
from pippin.snana_sim import SNANASimulation
from pippin.task import Task


class SNANALightCurveFit(ConfigBasedExecutable):
    """

    OUTPUTS:
    ========
        name: name given in the yml
        output_dir: top level output directory
        fitres_dirs: dirs containing the output fitres files. Is a list
        nml_file: location to copied nml file
        genversion: simulation genversion run against
        sim_name: name of the underlying sim task
        lc_output_dir: directory which contains fitres_dir and the simlogs dir
        fitopt_map: map from fitopt name (DEFAULT being nothing) to the FITOPTxxx.FITRES file
        is_data: true if the dependence is a DataPrep, false if its from a simulation
        blind: bool - whether or not to blind cosmo results
    """

    def __init__(self, name, output_dir, sim_task, config, global_config):
        self.data_dir = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../data_files/")

        self.config = config
        self.global_config = global_config

        base = config["BASE"]
        self.base_file = get_data_loc(self.data_dir, base)

        super().__init__(name, output_dir, self.base_file, " = ", dependencies=[sim_task])

        self.sim_task = sim_task
        self.sim_version = sim_task.output["genversion"]
        self.config_path = self.output_dir + "/FIT_" + self.sim_version + ".nml"
        self.lc_output_dir = f"{self.output_dir}/output"
        self.fitres_dirs = [os.path.join(self.lc_output_dir, os.path.basename(s)) for s in self.sim_task.output["sim_folders"]]

        self.logging_file = self.config_path.replace(".nml", ".nml_log")
        self.done_file = f"{self.output_dir}/FINISHED.DONE"
        secondary_log = f"{self.lc_output_dir}/SPLIT_JOBS_LCFIT/MERGELOGS/MERGE2.LOG"

        self.log_files = [self.logging_file, secondary_log]
        self.num_empty_threshold = 25  # Damn that tarball creation can be so slow
        self.output["fitres_dirs"] = self.fitres_dirs
        self.output["nml_file"] = self.config_path
        self.output["genversion"] = self.sim_version
        self.output["sim_name"] = sim_task.output["name"]
        self.output["blind"] = sim_task.output["blind"]
        self.output["lc_output_dir"] = self.lc_output_dir
        self.str_pattern = re.compile("[A-EG-SU-Za-eg-su-z]")

        is_data = False
        for d in self.dependencies:
            if isinstance(d, DataPrep):
                is_data = not d.output["is_sim"]
        self.output["is_data"] = is_data

        # Loading fitopts
        fitopts = config.get("FITOPTS", [])
        if isinstance(fitopts, str):
            fitopts = [fitopts]

        self.logger.debug("Loading fitopts")
        self.fitopts = []
        for f in fitopts:
            potential_path = get_data_loc(self.data_dir, f)
            if os.path.exists(potential_path):
                self.logger.debug(f"Loading in fitopts from {potential_path}")
                with open(potential_path) as f:
                    new_fitopts = list(f.read().splitlines())
                    self.fitopts += new_fitopts
                    self.logger.debug(f"Loaded {len(new_fitopts)} fitopts file from {potential_path}")
            else:
                assert "[" in f and "]" in f, f"Manual fitopt {f} for lcfit {self.name} should specify a label in square brackets"
                if not f.startswith("FITOPT:"):
                    f = "FITOPT: " + f
                self.logger.debug(f"Adding manual fitopt {f}")
                self.fitopts.append(f)
        # Map the fitopt outputs
        mapped = {"DEFAULT": "FITOPT000.FITRES"}
        for i, line in enumerate(self.fitopts):
            label = line.split("[")[1].split("]")[0]
            mapped[line] = f"FITOPT{i + 1:3d}.FITRES"
        self.output["fitopt_map"] = mapped
        self.output["fitres_file"] = os.path.join(self.fitres_dirs[0], mapped["DEFAULT"])

        self.options = self.config.get("OPTS", {})
        # Try to determine how many jobs will be put in the queue
        try:
            property = self.options.get("BATCH_INFO") or self.get_property("BATCH_INFO", assignment=": ")
            self.num_jobs = int(property.split()[-1])
        except Exception:
            self.num_jobs = 10

    def get_sim_dependency(self):
        for t in self.dependencies:
            if isinstance(t, SNANASimulation) or isinstance(t, DataPrep):
                return t.output
        return None

    def print_stats(self):
        folders = [f for f in os.listdir(self.lc_output_dir) if f.startswith("PIP_") and os.path.isdir(self.lc_output_dir + "/" + f)]
        if len(folders) > 5:
            self.logger.debug(f"Have {len(folders)} folders, only showing first five!")
            folders = folders[:5]
        for f in folders:
            path = os.path.join(self.lc_output_dir, f)
            try:
                full_path = os.path.join(path, "FITOPT000.FITRES")
                if not os.path.exists(full_path):
                    self.logger.info(f"{full_path} not found, seeing if it was gzipped")
                    full_path += ".gz"
                data = pd.read_csv(full_path, delim_whitespace=True, comment="#", compression="infer")
                d = data.groupby("TYPE").agg(num=("CID", "count"))
                self.logger.info("Types:  " + ("  ".join([f"{k}:{v}" for k, v in zip(d.index, d["num"].values)])))
                d.to_csv(os.path.join(path, "stats.txt"))
            except Exception:
                self.logger.error(f"Cannot load {path}")
                return False
        return True

    def set_snlcinp(self, name, value):
        """ Ensures the property name value pair is set in the SNLCINP section.

        Parameters
        ----------
        name : str
            The name of the property. Case insensitive, will be cast to upper.
        value : object
            The value to use. Object will be cast to string. For strings, include single quotes.
        """
        value = self.ensure_quotes_good(value)
        self.set_property(name, value, section_start="&SNLCINP", section_end="&END")

    def set_fitinp(self, name, value):
        """ Ensures the property name value pair is set in the FITINP section.

        Parameters
        ----------
        name : str
            The name of the property. Case insensitive, will be cast to upper.
        value : object
            The value to use. Object will be cast to string. For strings, include single quotes.
        """
        value = self.ensure_quotes_good(value)
        self.set_property(name, value, section_start="&FITINP", section_end="&END")

    def ensure_quotes_good(self, value):
        if isinstance(value, str):
            value = value.strip()
            if "!" in value:
                s = value.split("!", maxsplit=1)
                test = s[0].strip()
                comment = " ! " + s[1].strip()
            else:
                test = value
                comment = ""
            if self.str_pattern.search(test):
                if test.lower() not in [".true.", ".false."]:
                    # Ensure we have quotes
                    if test[0] == "'" and test[-1] == "'":
                        return value
                    else:
                        return f"'{test}'{comment}"
        return value

    def write_nml(self, force_refresh):

        # Parse config, first SNLCINP and then FITINP
        for key, value in self.config.get("SNLCINP", {}).items():
            self.set_snlcinp(key, value)
        for key, value in self.config.get("FITINP", {}).items():
            self.set_fitinp(key, value)
        for key, value in self.options.items():
            self.set_property(key, value, assignment=": ", section_end="&SNLCINP")

        if self.sim_task.output["ranseed_change"]:
            self.set_property("VERSION", self.sim_version + "-0*", assignment=": ", section_end="&SNLCINP")
        else:
            self.set_property("VERSION", self.sim_version, assignment=": ", section_end="&SNLCINP")

        self.set_property("OUTDIR", self.lc_output_dir, assignment=": ", section_end="&SNLCINP")
        self.set_property("DONE_STAMP", "FINISHED.DONE", assignment=": ", section_end="&SNLCINP")

        if isinstance(self.sim_task, DataPrep):
            self.set_snlcinp("PRIVATE_DATA_PATH", f"'{self.sim_task.output['data_path']}'")
            self.set_snlcinp("VERSION_PHOTOMETRY", f"'{self.sim_task.output['genversion']}'")

        # We want to do our hashing check here
        string_to_hash = self.fitopts + self.base
        # with open(os.path.abspath(inspect.stack()[0][1]), "r") as f:
        #     string_to_hash += f.read()
        new_hash = self.get_hash_from_string("".join(string_to_hash))
        old_hash = self.get_old_hash()
        regenerate = force_refresh or (old_hash is None or old_hash != new_hash)

        if regenerate:
            self.logger.info(f"Running Light curve fit. Removing output_dir")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            # Write main file
            with open(self.config_path, "w") as f:
                f.writelines(map(lambda s: s + "\n", string_to_hash))
            self.logger.info(f"NML file written to {self.config_path}")
            self.save_new_hash(new_hash)
            chown_dir(self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")

        return regenerate, new_hash

    def _run(self, force_refresh):
        regenerate, new_hash = self.write_nml(force_refresh)
        if not regenerate:
            self.should_be_done()
            return True
        self.logger.info(f"Light curve fitting outputting to {self.logging_file}")
        with open(self.logging_file, "w") as f:
            subprocess.run(["split_and_fit.pl", self.config_path, "NOPROMPT"], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
        return True

    def _check_completion(self, squeue):
        # Check for errors
        output_error = False
        for file in self.log_files:
            if os.path.exists(file):
                with open(file, "r") as f:
                    for line in f.read().splitlines():
                        if ("ABORT ON FATAL ERROR" in line or "QOSMaxSubmitJobPerUserLimit" in line) and not output_error:
                            extra = "" if squeue is None else f"Squeue shows you have {len(squeue)} jobs..."
                            self.logger.error(f"Fatal error in light curve fitting. See {file} for details. {extra}")
                            output_error = True
                        if output_error:
                            self.logger.error(f"Excerpt: {line}")

        if output_error:
            return Task.FINISHED_FAILURE

        # Check for existence of SPLIT_JOBS_LCFIT.tar.gz to see if job is done
        if os.path.exists(self.done_file):
            self.logger.info("Light curve done file found")
            logging_file2 = self.logging_file.replace("_log", "_log2")
            if not os.path.exists(logging_file2):
                self.logger.info(f"{logging_file2} not found, checking FITOPT existence")
                success = self.print_stats()
                if not success:
                    return Task.FINISHED_FAILURE
            return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, os.path.basename(self.config_path)[:-4])

    @staticmethod
    def get_tasks(config, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        tasks = []
        sim_tasks = Task.get_task_of_type(prior_tasks, DataPrep, SNANASimulation)
        for fit_name in config.get("LCFIT", []):
            num_matches = 0
            fit_config = config["LCFIT"][fit_name]
            mask = fit_config.get("MASK", "")
            if isinstance(mask, (str, int)):
                mask = [mask]
            for sim in sim_tasks:
                matches = np.any([m in sim.name] for m in mask)
                if matches:
                    num_matches += 1
                    fit_output_dir = f"{base_output_dir}/{stage_number}_LCFIT/{fit_name}_{sim.name}"
                    f = SNANALightCurveFit(f"{fit_name}_{sim.name}", fit_output_dir, sim, fit_config, global_config)
                    Task.logger.info(f"Creating fitting task {fit_name} with {f.num_jobs} jobs, for simulation {sim.name}")
                    tasks.append(f)
            if num_matches == 0:
                Task.logger.warning(f"LCFIT task {fit_name} with mask '{mask}' matched no sim_names: {[sim.name for sim in sim_tasks]}")
        return tasks
