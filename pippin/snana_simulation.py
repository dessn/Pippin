import os
import inspect
import logging
import shutil
import subprocess
import time

from pippin.base import ConfigBasedExecutable


class SNANASimulation(ConfigBasedExecutable):
    def __init__(self, output_dir, genversion, config, global_config, combine="combine.input"):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"

        super().__init__(output_dir, self.data_dir + combine, ":")

        self.genversion = genversion
        self.set_property("GENVERSION", genversion, assignment=":", section_end="ENDLIST_GENVERSION")
        self.config_path = self.output_dir + "/" + genversion + ".input"
        self.base_ia = config["IA"]["BASE"]
        self.base_cc = config["NONIA"]["BASE"]
        self.global_config = global_config

        # TODO: Would be good if in future versions we checked for changes and didnt delete if we didn't have to
        # TODO: The correct way of doing this is to read in all inputs, hash them, and save the hash as a file
        # Clean output dir. God I feel dangerous doing this, so hopefully unnecessary check
        if "//" not in self.output_dir and "Pippin" in self.output_dir:
            self.logger.debug(f"Cleaning output directory {self.output_dir}")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            os.makedirs(self.output_dir, exist_ok=True)

        shutil.copy(self.data_dir + self.base_ia, self.output_dir)
        shutil.copy(self.data_dir + self.base_cc, self.output_dir)

        # Copy the include input file if there is one
        with open(self.data_dir + self.base_ia, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("INPUT_FILE_INCLUDE"):
                    include_file = line.split(":")[-1].strip()
                    self.logger.debug(f"Copying included file {include_file}")
                    shutil.copy(self.data_dir + include_file, self.output_dir)

        for key in config.get("IA", []):
            if key.upper() == "BASE":
                continue
            self.set_property("GENOPT(Ia)", f"{key} {config['IA'][key]}", section_end="ENDLIST_GENVERSION")
        for key in config.get("NONIA", []):
            if key.upper() == "BASE":
                continue
            self.set_property("GENOPT(NON1A)", f"{key} {config['NONIA'][key]}", section_end="ENDLIST_GENVERSION")

        for key in config.get("GLOBAL", []):
            if key.upper() == "BASE":
                continue
            self.set_property(key, config['GLOBAL'][key])

        self.set_property("SIMGEN_INFILE_Ia", self.base_ia)
        self.set_property("SIMGEN_INFILE_NONIa", self.base_cc)
        self.set_property("GENPREFIX", self.genversion)

    def write_input(self):
        with open(self.config_path, "w") as f:
            f.writelines(map(lambda s: s + '\n', self.base))
        self.logger.info(f"Input file written to {self.config_path}")

    def run(self):
        self.write_input()
        logging_file = self.config_path.replace(".input", ".input_log")
        with open(logging_file, "w") as f:
            subprocess.run(["sim_SNmix.pl", self.config_path], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)

        self.logger.info(f"Sim logging outputting to {logging_file}")
        done_file = f"{self.output_dir}/SIMLOGS_{self.genversion}/SIMJOB_ALL.DONE"

        # Monitor for success or failure
        while True:
            time.sleep(self.global_config["OUTPUT"].getint("ping_frequency"))

            # Check log for errors and if found, print the rest of the log so you dont have to look up the file
            with open(logging_file, "r") as f:
                output_error = False
                for line in f.read().splitlines():
                    if "ERROR" in line:
                        self.logger.critical(f"Fatal error in simulation. See {logging_file} for details.")
                        output_error = True
                    if output_error:
                        self.logger.error(f"Excerpt: {line}")
            if output_error:
                return False

            # Check to see if the done file exists
            if os.path.exists(done_file):
                sim_folder = f"{self.global_config['SNANA']['sim_dir']}/{self.genversion}"
                sim_folder_endpoint = f"{self.output_dir}/{self.genversion}"
                self.logger.info("Done file found, creating symlinks")
                self.logger.debug(f"Linking {sim_folder} -> {sim_folder_endpoint}")
                os.symlink(sim_folder, sim_folder_endpoint, target_is_directory=True)
                return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    s = SNANASimulation("test", "testv")

    s.set_property("TESTPROP", "HELLO")
    s.delete_property("GOODBYE")
    s.write_input()

