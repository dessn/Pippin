import os
import shutil
import subprocess
import re
import pandas as pd

from pippin.base import ConfigBasedExecutable
from pippin.config import mkdirs, get_data_loc, chown_dir, read_yaml
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

        self.config = config
        self.global_config = global_config

        base = config.get("BASE")
        if base is None:
            Task.fail_config(f"You have not specified a BASE nml file for task {name}")
        self.base_file = get_data_loc(base)
        if self.base_file is None:
            Task.fail_config(f"Base file {base} cannot be found for task {name}")

        self.convert_base_file()
        super().__init__(name, output_dir, config, self.base_file, " = ", dependencies=[sim_task])

        self.sim_task = sim_task
        self.sim_version = sim_task.output["genversion"]
        self.config_path = self.output_dir + "/FIT_" + self.sim_version + ".nml"
        self.lc_output_dir = os.path.join(self.output_dir, "output")
        self.lc_log_dir = os.path.join(self.lc_output_dir, "SPLIT_JOBS_LCFIT")
        self.fitres_dirs = [os.path.join(self.lc_output_dir, os.path.basename(s)) for s in self.sim_task.output["sim_folders"]]

        self.logging_file = self.config_path.replace(".nml", ".LOG")
        self.kill_file = self.config_path.replace(".input", "_KILL.LOG")

        self.done_file = f"{self.lc_output_dir}/ALL.DONE"

        self.merge_log = os.path.join(self.lc_output_dir, "MERGE.LOG")

        self.log_files = [self.logging_file]
        self.num_empty_threshold = 20  # Damn that tarball creation can be so slow
        self.display_threshold = 8
        self.output["fitres_dirs"] = self.fitres_dirs
        self.output["base_file"] = self.base_file
        self.output["nml_file"] = self.config_path
        self.output["genversion"] = self.sim_version
        self.output["sim_name"] = sim_task.output["name"]
        self.output["blind"] = sim_task.output["blind"]
        self.output["lc_output_dir"] = self.lc_output_dir
        self.str_pattern = re.compile("[A-DG-SU-Za-dg-su-z]")

        self.validate_fitopts(config)

        is_data = False
        for d in self.dependencies:
            if isinstance(d, DataPrep):
                is_data = not d.output["is_sim"]
        self.output["is_data"] = is_data

        self.options = self.config.get("OPTS", {})
        # Try to determine how many jobs will be put in the queue
        try:
            property = self.options.get("BATCH_INFO") or self.yaml["CONFIG"].get("BATCH_INFO")
            self.num_jobs = int(property.split()[-1])
        except Exception:
            self.logger.warning("Could not determine BATCH_INFO for job, setting num_jobs to 10")
            self.num_jobs = 10

    def validate_fitopts(self, config):
        # Loading fitopts
        fitopts = config.get("FITOPTS", [])
        if isinstance(fitopts, str):
            fitopts = [fitopts]

        self.logger.debug("Loading fitopts")

        self.raw_fitopts = []
        for f in fitopts:
            self.logger.debug(f"Parsing fitopt {f}")
            potential_path = get_data_loc(f)
            if potential_path is not None and os.path.exists(potential_path):
                self.logger.debug(f"Loading in fitopts from {potential_path}")
                y = read_yaml(potential_path)
                assert isinstance(y, dict), "New FITOPT format for external files is a yaml dictionary. See global.yml for an example."
                self.raw_fitopts.append(y)
                self.logger.debug(f"Loaded a fitopt dictionary file from {potential_path}")
            else:
                assert f.strip().startswith(
                    "/"
                ), f"Manual fitopt {f} for lcfit {self.name} should specify a label wrapped with /. If this is meant to be a file, it doesnt exist."
                self.logger.debug(f"Adding manual fitopt {f}")
                self.raw_fitopts.append(f)

    def compute_fitopts(self):
        """ Runs after the sim/data to locate the survey """

        survey = self.get_sim_dependency()["SURVEY"]

        # Determine final fitopts based on survey
        fitopts = []
        for f in self.raw_fitopts:
            if isinstance(f, str):
                fitopts.append(f)
                self.logger.debug(f"Adding manual fitopt: {f}")
            elif isinstance(f, dict):
                for key, values in f.items():
                    if key in ["GLOBAL", survey]:
                        assert isinstance(values, dict), "Fitopt values should be a dict of label: scale command"

                        for label, scale_command in values.items():
                            scale, command = scale_command.split(maxsplit=1)
                            fitopt = f"/{label}/ {command}"
                            self.logger.debug(f"Adding FITOPT from {key}: {fitopt}")
                            fitopts.append(fitopt)
            else:
                raise ValueError(f"Fitopt item {f} is not a string or dictionary, what on earth is it?")

        # Map the fitopt outputs
        mapped = {"DEFAULT": "FITOPT000.FITRES.gz"}
        mapped2 = {0: "DEFAULT"}
        for i, line in enumerate(fitopts):
            label = line.strip().split("/")[1]
            mapped[label] = f"FITOPT{i + 1:03d}.FITRES.gz"
            mapped2[i] = label
        if fitopts:
            self.yaml["CONFIG"]["FITOPT"] = fitopts
        self.output["fitopt_map"] = mapped
        self.output["fitopt_index"] = mapped2
        self.output["fitres_file"] = os.path.join(self.fitres_dirs[0], mapped["DEFAULT"])

    def convert_base_file(self):
        self.logger.debug(f"Translating base file {self.base_file}")
        try:
            subprocess.run(["submit_batch_jobs.sh", "--opt_translate", "10", os.path.basename(self.base_file)], cwd=os.path.dirname(self.base_file))
        except FileNotFoundError:
            # For testing, this wont exist
            pass

    def get_sim_dependency(self):
        for t in self.dependencies:
            if isinstance(t, SNANASimulation) or isinstance(t, DataPrep):
                return t.output
        return None

    def print_stats(self):
        folders = [f for f in os.listdir(self.lc_output_dir) if os.path.isdir(self.lc_output_dir + "/" + f)]
        if len(folders) > 5:
            self.logger.debug(f"Have {len(folders)} folders, only showing first five!")
            folders = folders[:5]
        for f in folders:
            path = os.path.join(self.lc_output_dir, f)
            try:
                full_path = os.path.join(path, "FITOPT000.FITRES.gz")
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
            self.yaml["CONFIG"][key] = value

        self.compute_fitopts()

        if self.sim_task.output["ranseed_change"]:
            self.yaml["CONFIG"]["VERSION"] = [self.sim_version + "-0*"]
        else:
            self.yaml["CONFIG"]["VERSION"] = [self.sim_version]

        self.yaml["CONFIG"]["OUTDIR"] = self.lc_output_dir
        # self.yaml["CONFIG"]["DONE_STAMP"] = "ALL.DONE"

        if isinstance(self.sim_task, DataPrep):
            data_path = self.sim_task.output["data_path"]
            if "SNDATA_ROOT/lcmerge" not in data_path:
                self.set_snlcinp("PRIVATE_DATA_PATH", f"'{self.sim_task.output['data_path']}'")
            self.set_snlcinp("VERSION_PHOTOMETRY", f"'{self.sim_task.output['genversion']}'")

        # We want to do our hashing check here
        string_to_hash = self.get_output_string()
        new_hash = self.get_hash_from_string(string_to_hash)
        old_hash = self.get_old_hash()
        regenerate = force_refresh or (old_hash is None or old_hash != new_hash)

        if regenerate:
            self.logger.info(f"Running Light curve fit. Removing output_dir")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            # Write main file

            # Write the primary input file
            self.write_output_file(self.config_path)
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
            subprocess.run(["submit_batch_jobs.sh", os.path.basename(self.config_path)], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
        return True

    def kill_and_fail(self):
        with open(self.kill_file, "w") as f:
            self.logger.info(f"Killing remaining jobs for {self.name}")
            subprocess.run(["submit_batch_jobs.sh", "--kill", os.path.basename(self.config_path)], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
        return Task.FINISHED_FAILURE

    def check_issues(self):
        log_files = [] + self.log_files
        log_files += [os.path.join(self.lc_log_dir, f) for f in os.listdir(self.lc_log_dir) if f.upper().endswith(".LOG")]

        self.scan_files_for_error(log_files, "FATAL ERROR ABORT", "QOSMaxSubmitJobPerUserLimit", "DUE TO TIME LIMIT")
        return Task.FINISHED_FAILURE

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.info("Light curve done file found")
            if not os.path.exists(self.logging_file):
                self.logger.info(f"{self.logging_file} not found, checking FITOPT existence")
                success = self.print_stats()
                if not success:
                    return Task.FINISHED_FAILURE

            with open(self.done_file) as f:
                if "SUCCESS" in f.read():
                    if os.path.exists(self.merge_log):
                        y = read_yaml(self.merge_log)
                        if "MERGE" in y.keys():
                            for i, row in enumerate(y["MERGE"]):
                                state, iver, fitopt, n_all, n_snanacut, n_fitcut, cpu = row
                                if cpu < 60:
                                    units = "minutes"
                                else:
                                    cpu = cpu / 60
                                    units = "hours"
                                self.logger.info(
                                    f"LCFIT {i + 1} fit {n_all} events. {n_snanacut} passed SNANA cuts, {n_fitcut} passed fitcuts, taking {cpu:0.1f} CPU {units}"
                                )
                        else:
                            self.logger.error(f"File {self.merge_log} does not have a MERGE section - did it die?")
                            return Task.FINISHED_FAILURE
                        if "SURVEY" in y.keys():
                            self.output["SURVEY"] = y["SURVEY"]
                            self.output["SURVEY_ID"] = y["IDSURVEY"]
                        else:
                            s = self.get_sim_dependency()
                            self.output["SURVEY"] = s["SURVEY"]
                            self.output["SURVEY_ID"] = s["SURVEY_ID"]
                        return Task.FINISHED_SUCCESS
                    else:
                        return Task.FINISHED_FAILURE
                else:
                    self.logger.debug(f"Done file reporting failure, scanning log files in {self.lc_log_dir}")
                    return self.check_issues()
        elif not os.path.exists(self.merge_log):
            self.logger.error("MERGE.LOG was not created, job died on submission")
            return self.check_issues()

        return self.check_for_job(squeue, os.path.basename(self.config_path))

    @staticmethod
    def get_tasks(config, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        tasks = []

        all_deps = Task.match_tasks_of_type(None, prior_tasks, DataPrep, SNANASimulation)
        for fit_name in config.get("LCFIT", []):
            num_matches = 0
            fit_config = config["LCFIT"][fit_name]
            mask = fit_config.get("MASK", "")

            sim_tasks = Task.match_tasks_of_type(mask, prior_tasks, DataPrep, SNANASimulation)
            for sim in sim_tasks:
                num_matches += 1
                fit_output_dir = f"{base_output_dir}/{stage_number}_LCFIT/{fit_name}_{sim.name}"
                f = SNANALightCurveFit(f"{fit_name}_{sim.name}", fit_output_dir, sim, fit_config, global_config)
                Task.logger.info(f"Creating fitting task {fit_name} with {f.num_jobs} jobs, for simulation {sim.name}")
                tasks.append(f)
            if num_matches == 0:
                Task.fail_config(f"LCFIT task {fit_name} with mask '{mask}' matched no sim_names: {[sim.name for sim in all_deps]}")
        return tasks
