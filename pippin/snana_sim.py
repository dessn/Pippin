import os
import inspect
import logging
import shutil
import subprocess
import tempfile
import collections
import json

from pippin.base import ConfigBasedExecutable
from pippin.config import chown_dir, copytree, mkdirs, get_data_loc, get_hash
from pippin.task import Task


class SNANASimulation(ConfigBasedExecutable):
    """ Merge fitres files and aggregator output

    CONFIGURATION:
    ==============
    SIM:
      label:
        IA_name:  # Must have a sim type that starts with IA_ or is called exactly IA. Its how it gets divided into Ia or contaminant.
          BASE: sn_ia_salt2_g10_des3yr.input # Location of an input file. Either in the data_files dir or full path
          DNDZ_ALLSCALE: 3.0  # Override input file content here
        II_JONES:
          BASE: sn_collection_jones.input
        GLOBAL:
            NGEN_UNIT: 1  # Set properties for all input file shere


    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        genversion: genverison of sim
        types_dict: dict map from IA or NONIA to numeric gentypes
        types: dict map from numeric gentype to string (Ia, II, etc)
        photometry_dirs: location of fits files with photometry. is a list.
        ranseed_change: true or false for if RANSEED_CHANGE was set
        blind: bool - whether to blind cosmo results
    """

    def __init__(self, name, output_dir, genversion, config, global_config, combine="combine.input"):
        self.data_dir = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../data_files/")
        super().__init__(name, output_dir, self.data_dir + combine, ": ")

        self.genversion = genversion
        if len(genversion) < 30:
            self.genprefix = self.genversion
        else:
            hash = get_hash(self.genversion)[:5]
            self.genprefix = self.genversion[:25] + hash

        self.config = config
        self.options = config.get("OPTS", {})
        self.reserved_keywords = ["BASE"]
        self.config_path = f"{self.output_dir}/{self.genversion}.input"  # Make sure this syncs with the tmp file name
        self.base_ia = [config[k]["BASE"] for k in config.keys() if k.startswith("IA_") or k == "IA"]
        self.base_cc = [config[k]["BASE"] for k in config.keys() if not k.startswith("IA_") and k != "IA" and k != "GLOBAL"]
        self.global_config = global_config

        rankeys = [r for r in config["GLOBAL"].keys() if r.startswith("RANSEED_")]
        value = int(config["GLOBAL"][rankeys[0]].split(" ")[0]) if rankeys else 1
        self.set_num_jobs(2 * value)

        self.sim_log_dir = f"{self.output_dir}/LOGS"
        self.total_summary = os.path.join(self.sim_log_dir, "TOTAL_SUMMARY.LOG")
        self.done_file = f"{self.output_dir}/FINISHED.DONE"
        self.logging_file = self.config_path.replace(".input", ".input_log")
        self.output["blind"] = self.options.get("BLIND", False)

        self.derived_batch_info = None
        # Try to determine how many jobs will be put in the queue
        try:
            # If BATCH_INFO is set, we'll use that
            batch_info = self.config.get("GLOBAL", {}).get("BATCH_INFO")
            default_batch_info = self.get_property("BATCH_INFO", assignment=": ")

            # If its not set, lets check for ranseed_repeat or ranseed_change
            if batch_info is None:
                ranseed_repeat = self.config.get("GLOBAL", {}).get("RANSEED_REPEAT")
                ranseed_change = self.config.get("GLOBAL", {}).get("RANSEED_CHANGE")
                ranseed = ranseed_repeat or ranseed_change

                if ranseed:
                    num_jobs = int(ranseed.strip().split()[0])
                    self.logger.debug(f"Found a randseed with {num_jobs}, deriving batch info")
                    comps = default_batch_info.strip().split()
                    comps[-1] = str(num_jobs)
                    self.derived_batch_info = " ".join(comps)
                    self.num_jobs = num_jobs
            else:
                # self.logger.debug(f"BATCH INFO property detected as {property}")
                self.num_jobs = int(default_batch_info.split()[-1])
        except Exception:
            self.logger.warning(f"Unable to determine how many jobs simulation {self.name} has")
            self.num_jobs = 10

        self.output["genversion"] = self.genversion
        self.output["genprefix"] = self.genprefix

        ranseed_change = self.config.get("GLOBAL", {}).get("RANSEED_CHANGE")
        base = os.path.expandvars(f"{self.global_config['SNANA']['sim_dir']}/{self.genversion}")
        if ranseed_change:
            num_sims = int(ranseed_change.split()[0])
            self.sim_folders = [base + f"-{i + 1:04d}" for i in range(num_sims)]
        else:
            self.sim_folders = [base]
        self.output["ranseed_change"] = ranseed_change is not None
        self.output["sim_folders"] = self.sim_folders

    def write_input(self, force_refresh):
        self.set_property("GENVERSION", self.genversion, assignment=": ", section_end="ENDLIST_GENVERSION")
        self.set_property("LOGDIR", os.path.basename(self.sim_log_dir), assignment=": ", section_end="ENDLIST_GENVERSION")
        for k in self.config.keys():
            if k.upper() != "GLOBAL":
                run_config = self.config[k]
                run_config_keys = list(run_config.keys())
                assert "BASE" in run_config_keys, "You must specify a base file for each option"
                for key in run_config_keys:
                    if key.upper() in self.reserved_keywords:
                        continue
                    base_file = run_config["BASE"]
                    match = os.path.basename(base_file).split(".")[0]
                    val = run_config[key]
                    if not isinstance(val, list):
                        val = [val]
                    for v in val:
                        self.set_property(f"GENOPT({match})", f"{key} {v}", section_end="ENDLIST_GENVERSION", only_add=True)

        for key in self.config.get("GLOBAL", []):
            if key.upper() == "BASE":
                continue
            direct_set = ["FORMAT_MASK", "RANSEED_REPEAT", "RANSEED_CHANGE", "BATCH_INFO", "BATCH_MEM", "NGEN_UNIT", "RESET_CIDOFF"]
            if key in direct_set:
                self.set_property(key, self.config["GLOBAL"][key], assignment=": ")
            else:
                self.set_property(f"GENOPT_GLOBAL: {key}", self.config["GLOBAL"][key], assignment=" ")

            if self.derived_batch_info:
                self.set_property("BATCH_INFO", self.derived_batch_info, assignment=": ")

            if key == "RANSEED_CHANGE":
                self.delete_property("RANSEED_REPEAT")
            elif key == "RANSEED_REPEAT":
                self.delete_property("RANSEED_CHANGE")

        self.set_property("SIMGEN_INFILE_Ia", " ".join([os.path.basename(f) for f in self.base_ia]) if self.base_ia else None)
        self.set_property("SIMGEN_INFILE_NONIa", " ".join([os.path.basename(f) for f in self.base_cc]) if self.base_cc else None)
        self.set_property("GENPREFIX", self.genprefix)

        # Put config in a temp directory
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        # Copy the base files across
        for f in self.base_ia:
            shutil.copy(get_data_loc(self.data_dir, f), temp_dir)
        for f in self.base_cc:
            shutil.copy(get_data_loc(self.data_dir, f), temp_dir)

        # Copy the include input file if there is one
        input_copied = []
        fs = self.base_ia + self.base_cc
        for ff in fs:
            if ff not in input_copied:
                input_copied.append(ff)
                path = ff if ff.startswith("/") else os.path.join(self.data_dir, ff)
                with open(path, "r") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line.startswith("INPUT_FILE_INCLUDE"):
                            include_file = line.split(":")[-1].strip()
                            self.logger.debug(f"Copying included file {include_file}")
                            if include_file.startswith("/"):
                                shutil.copy(include_file, temp_dir)
                            else:
                                include_file = get_data_loc(self.data_dir, include_file)
                                # Dont copy it over, we need to sed it to update the INPUT_FILE_INCLUDE to be relative
                                # Ah crap, this will only work for a single include.
                                base = os.path.basename(include_file)
                                input_file = os.path.join(temp_dir, os.path.basename(ff))
                                sed_command = f"sed -i -e 's|{include_file}|{base}|g' {input_file}"
                                self.logger.debug(f"Running sed command: {sed_command}")
                                subprocess.run(sed_command, stderr=subprocess.STDOUT, cwd=temp_dir, shell=True)

                                shutil.copy(self.data_dir + include_file, temp_dir)

                            fs.append(os.path.join(temp_dir, os.path.basename(include_file)))

        # Write the primary input file
        main_input_file = f"{temp_dir}/{self.genversion}.input"
        with open(main_input_file, "w") as f:
            f.writelines(map(lambda s: s + "\n", self.base))
        self.logger.info(f"Input file written to {main_input_file}")

        # Remove any duplicates and order the output files
        output_files = [f"{temp_dir}/{a}" for a in sorted(os.listdir(temp_dir))]
        self.logger.debug(f"{len(output_files)} files used to create simulation. Hashing them.")

        # Get current hash
        new_hash = self.get_hash_from_files(output_files)
        old_hash = self.get_old_hash()
        regenerate = force_refresh or (old_hash is None or old_hash != new_hash)

        if regenerate:
            self.logger.info(f"Running simulation")
            # Clean output dir. God I feel dangerous doing this, so hopefully unnecessary check
            if "//" not in self.output_dir and len(self.output_dir) > 30:
                self.logger.debug(f"Cleaning output directory {self.output_dir}")
                shutil.rmtree(self.output_dir, ignore_errors=True)
                mkdirs(self.output_dir)
                self.logger.debug(f"Copying from {temp_dir} to {self.output_dir}")
                copytree(temp_dir, self.output_dir)
                self.save_new_hash(new_hash)
            else:
                self.logger.error(f"Seems to be an issue with the output dir path: {self.output_dir}")

            chown_dir(self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")
        temp_dir_obj.cleanup()
        return regenerate, new_hash

    def _run(self, force_refresh):

        regenerate, new_hash = self.write_input(force_refresh)
        if not regenerate:
            self.should_be_done()
            return True

        with open(self.logging_file, "w") as f:
            subprocess.run(["sim_SNmix.pl", self.config_path], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)

        self.logger.info(f"Sim running and logging outputting to {self.logging_file}")
        return True

    def _check_completion(self, squeue):
        # Check log for errors and if found, print the rest of the log so you dont have to look up the file
        output_error = False
        if self.logging_file is not None and os.path.exists(self.logging_file):
            with open(self.logging_file, "r") as f:
                for line in f.read().splitlines():
                    if "ABORT ON FATAL ERROR" in line:
                        self.logger.error(f"Fatal error in simulation. See {self.logging_file} for details.")
                        output_error = True
                    if output_error:
                        self.logger.info(f"Excerpt: {line}")
            if output_error:
                self.logger.debug("Removing hash on failure")
                os.remove(self.hash_file)
                chown_dir(self.output_dir)
                return Task.FINISHED_FAILURE
        else:
            self.logger.warn(f"Simulation {self.name} logging file does not exist: {self.logging_file}")
        if os.path.exists(self.sim_log_dir):
            for file in os.listdir(self.sim_log_dir):
                if not file.startswith("TMP") or not file.endswith(".LOG"):
                    continue
                try:
                    with open(self.sim_log_dir + "/" + file, "r") as f:
                        for line in f.read().splitlines():
                            if (" ABORT " in line or "FATAL[" in line) and not output_error:
                                output_error = True
                                self.logger.error(f"Fatal error in simulation. See {self.sim_log_dir}/{file} for details.")
                            if output_error:
                                self.logger.info(f"Excerpt: {line}")
                    if output_error:
                        self.logger.debug("Removing hash on failure")
                        os.remove(self.hash_file)
                        chown_dir(self.output_dir)
                        return Task.FINISHED_FAILURE
                except FileNotFoundError:
                    self.logger.warning("Cannot find log file in process of reading it in (maybe it just got zipped up): {self.sim_log_dir + '/' + file}")
        else:
            self.logger.error(f"Sim log dir doesn't exist at {self.sim_log_dir}. Warning, there might be orphaned jobs from this.")
            return Task.FINISHED_FAILURE

        # Check to see if the done file exists

        if os.path.exists(self.done_file):
            self.logger.info(f"Simulation {self.name} found done file!")
            if os.path.exists(self.total_summary):
                with open(self.total_summary) as f:
                    key, count = None, None
                    allzero = True
                    for line in f.readlines():
                        if line.strip().startswith("SUM-"):
                            key = line.strip().split()[0]
                        if line.strip().startswith(self.genversion):
                            count = line.split()[2]
                            self.logger.debug(f"Simulation reports {key} wrote {count} to file")
                            if int(count.strip()) > 0:
                                allzero = False
                    if allzero:
                        self.logger.error(f"Simulation didn't write anything out according to {self.total_summary}")
                        return Task.FINISHED_FAILURE
            else:
                self.logger.warning(f"Cannot find {self.total_summary}")

            with open(self.done_file) as f:
                if "FAILURE" in f.read():
                    self.logger.error(f"Done file {self.done_file} reporting failure")
                    return Task.FINISHED_FAILURE

            self.logger.info("Done file found, creating symlinks")
            s_ends = [os.path.join(self.output_dir, os.path.basename(s)) for s in self.sim_folders]
            for s, s_end in zip(self.sim_folders, s_ends):
                if not os.path.exists(s_end):
                    self.logger.debug(f"Linking {s} -> {s_end}")
                    os.symlink(s, s_end, target_is_directory=True)
                chown_dir(self.output_dir)
            self.output.update({"photometry_dirs": s_ends, "types": self.get_types()})
            return Task.FINISHED_SUCCESS

        return self.check_for_job(squeue, f"{self.genprefix}_0")

    def resolve_name_to_type(self, name):
        """ I know this is wrong, but its just for Supernnova to split Ia and everything else """
        name = name.upper()
        if "SNIA_" in name or "SALT2" in name or "SN_IA_" in name:
            return "Ia"
        else:
            return "II"

    def get_types(self):
        types = {}
        for f in [f for f in os.listdir(self.output_dir) if f.endswith(".input")]:
            path = os.path.join(self.output_dir, f)
            name = f.split(".")[0]
            with open(path, "r") as file:
                for line in file.readlines():
                    if line.startswith("GENTYPE"):
                        num = int(line.split(":")[1].strip())
                        number = "1" + "%02d" % num
                        n = self.resolve_name_to_type(name)
                        types[number] = n
                        types[str(num)] = n
                        break
        sorted_types = collections.OrderedDict(sorted(types.items()))
        self.logger.debug(f"Types found: {json.dumps(sorted_types)}")
        types_dict = {"IA": [], "NONIA": []}
        for key, value in sorted_types.items():
            if value.upper() == "IA":
                types_dict["IA"].append(int(key))
            else:
                types_dict["NONIA"].append(int(key))
        self.output["types_dict"] = types_dict
        return sorted_types

    @staticmethod
    def get_tasks(config, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        tasks = []
        for sim_name in config.get("SIM", []):
            sim_output_dir = f"{base_output_dir}/{stage_number}_SIM/{sim_name}"
            s = SNANASimulation(sim_name, sim_output_dir, f"{prefix}_{sim_name}", config["SIM"][sim_name], global_config)
            Task.logger.debug(f"Creating simulation task {sim_name} with {s.num_jobs} jobs, output to {sim_output_dir}")
            tasks.append(s)
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    s = SNANASimulation("test", "testv")

    s.set_property("TESTPROP", "HELLO")
    s.delete_property("GOODBYE")
    s.write_input()
