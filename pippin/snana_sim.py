import os
import inspect
import logging
import shutil
import subprocess
import tempfile
import collections
import json

from pippin.base import ConfigBasedExecutable
from pippin.config import chown_dir, copytree, mkdirs
from pippin.task import Task


class SNANASimulation(ConfigBasedExecutable):
    def __init__(self, name, output_dir, genversion, config, global_config, combine="combine.input"):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        super().__init__(name, output_dir, self.data_dir + combine, ": ")

        self.genversion = genversion
        self.config = config
        self.reserved_keywords = ["BASE"]
        self.config_path = f"{self.output_dir}/{self.genversion}.input"  # Make sure this syncs with the tmp file name
        self.base_ia = [config[k]["BASE"] for k in config.keys() if k.startswith("IA_") or k == "IA"]
        self.base_cc = [config[k]["BASE"] for k in config.keys() if not k.startswith("IA_") and k != "IA" and k != "GLOBAL"]
        self.global_config = global_config

        rankeys = [r for r in config["GLOBAL"].keys() if r.startswith("RANSEED_")]
        value = int(config["GLOBAL"][rankeys[0]].split(" ")[0]) if rankeys else 1
        self.set_num_jobs(2 * value)

        self.sim_log_dir = f"{self.output_dir}/SIMLOGS_{self.genversion}"
        self.total_summary = os.path.join(self.sim_log_dir, "TOTAL_SUMMARY.LOG")
        self.done_file = f"{self.output_dir}/FINISHED.DONE"
        self.logging_file = self.config_path.replace(".input", ".input_log")

        self.output["genversion"] = self.genversion

    def write_input(self, force_refresh):
        self.set_property("GENVERSION", self.genversion, assignment=": ", section_end="ENDLIST_GENVERSION")
        for k in self.config.keys():
            if k.upper() != "GLOBAL":
                run_config = self.config[k]
                run_config_keys = list(run_config.keys())
                assert "BASE" in run_config_keys, "You must specify a base file for each option"
                for key in run_config_keys:
                    if key.upper() in self.reserved_keywords:
                        continue
                    base_file = run_config["BASE"]
                    match = base_file.split(".")[0]
                    self.set_property(f"GENOPT({match})", f"{key} {run_config[key]}", section_end="ENDLIST_GENVERSION")

        for key in self.config.get("GLOBAL", []):
            if key.upper() == "BASE":
                continue
            self.set_property(key, self.config['GLOBAL'][key])
            if key == "RANSEED_CHANGE":
                self.delete_property("RANSEED_REPEAT")
            elif key == "RANSEED_REPEAT":
                self.delete_property("RANSEED_CHANGE")

        self.set_property("SIMGEN_INFILE_Ia", " ".join(self.base_ia) if self.base_ia else None)
        self.set_property("SIMGEN_INFILE_NONIa", " ".join(self.base_cc) if self.base_cc else None)
        self.set_property("GENPREFIX", self.genversion)

        # Put config in a temp directory
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        # Copy the base files across
        for f in self.base_ia:
            shutil.copy(self.data_dir + f, temp_dir)
        for f in self.base_cc:
            shutil.copy(self.data_dir + f, temp_dir)

        # Copy the include input file if there is one
        input_copied = []
        fs = self.base_ia + self.base_cc
        for ff in fs:
            if ff not in input_copied:
                input_copied.append(ff)
                with open(self.data_dir + ff, "r") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line.startswith("INPUT_FILE_INCLUDE"):
                            include_file = line.split(":")[-1].strip()
                            self.logger.debug(f"Copying included file {include_file}")
                            shutil.copy(self.data_dir + include_file, temp_dir)

        # Write the primary input file
        main_input_file = f"{temp_dir}/{self.genversion}.input"
        with open(main_input_file, "w") as f:
            f.writelines(map(lambda s: s + '\n', self.base))
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
            return True

        with open(self.logging_file, "w") as f:
            subprocess.run(["sim_SNmix.pl", self.config_path], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
        shutil.chown(self.logging_file, group=self.global_config["SNANA"]["group"])

        self.logger.info(f"Sim running and logging outputting to {self.logging_file}")
        return True

    def _check_completion(self, squeue):
        # Check log for errors and if found, print the rest of the log so you dont have to look up the file
        output_error = False
        if self.logging_file is not None and os.path.exists(self.logging_file):
            with open(self.logging_file, "r") as f:
                for line in f.read().splitlines():
                    if "ERROR" in line or "***** ABORT *****" in line:
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
        for file in os.listdir(self.sim_log_dir):
            if not file.startswith("TMP") or not file.endswith(".LOG"):
                continue
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

        # Check to see if the done file exists
        sim_folder_endpoint = f"{self.output_dir}/{self.genversion}"
        if os.path.exists(self.done_file):
            self.logger.info(f"Simulation {self.name} found done file!")
            if os.path.exists(self.total_summary):
                with open(self.total_summary) as f:
                    key, count = None, None
                    for line in f.readlines():
                        if line.strip().startswith("SUM-"):
                            key = line.strip().split()[0]
                        if line.strip().startswith(self.genversion):
                            count = line.split()[2]
                            self.logger.debug(f"Simulation reports {key} wrote {count} to file")
            else:
                self.logger.debug(f"Cannot find {self.total_summary}")
            if not os.path.exists(sim_folder_endpoint):
                sim_folder = os.path.expandvars(f"{self.global_config['SNANA']['sim_dir']}/{self.genversion}")
                self.logger.info("Done file found, creating symlinks")
                self.logger.debug(f"Linking {sim_folder} -> {sim_folder_endpoint}")
                os.symlink(sim_folder, sim_folder_endpoint, target_is_directory=True)
                chown_dir(self.output_dir)
            self.output = {
                "photometry_dir": sim_folder_endpoint,
                "types": self.get_types(),
            }
            return Task.FINISHED_SUCCESS
        return 0  # TODO: Update to num jobs

    def get_types(self):
        types = {}
        for f in [f for f in os.listdir(self.output_dir) if f.endswith(".input")]:
            path = os.path.join(self.output_dir, f)
            name = f.split(".")[0]
            with open(path, "r") as file:
                for line in file.readlines():
                    if line.startswith("GENTYPE"):
                        number = "1" + "%02d" % int(line.split(":")[1].strip())
                        types[number] = name
                        break
        sorted_types = collections.OrderedDict(sorted(types.items()))
        self.logger.debug(f"Types found: {json.dumps(sorted_types)}")
        return sorted_types


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    s = SNANASimulation("test", "testv")

    s.set_property("TESTPROP", "HELLO")
    s.delete_property("GOODBYE")
    s.write_input()

