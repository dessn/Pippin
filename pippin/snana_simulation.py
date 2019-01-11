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

        self.hash = None

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
        # Load previous hash here if it exists
        old_hash = None
        hash_file = self.output_dir + "hash.md5"
        if os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                old_hash = int(f.read().strip())
                self.logger.debug(f"Previous result found, hash is {old_hash}")

        # Copy the base files across
        output_files = []
        shutil.copy(self.data_dir + self.base_ia, self.output_dir)
        output_files.append(self.output_dir + self.base_ia)
        shutil.copy(self.data_dir + self.base_cc, self.output_dir)
        output_files.append(self.output_dir + self.base_cc)

        # Copy the include input file if there is one
        with open(self.data_dir + self.base_ia, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("INPUT_FILE_INCLUDE"):
                    include_file = line.split(":")[-1].strip()
                    self.logger.debug(f"Copying included file {include_file}")
                    shutil.copy(self.data_dir + include_file, self.output_dir)
                    output_files.append(self.output_dir + include_file)

        # Write the primary input file
        with open(self.config_path, "w") as f:
            f.writelines(map(lambda s: s + '\n', self.base))
        output_files.append(self.config_path)
        self.logger.info(f"Input file written to {self.config_path}")

        # Remove any duplicates and order the output files
        output_files = sorted(list(set(output_files)))
        self.logger.debug(f"{len(output_files)} files used to create simulation. Hashing them.")

        # Also add this file to the hash, so if the code changes we also regenerate. Smart.
        output_files.append(os.path.abspath(inspect.stack()[0][1]))

        # Get current hash
        string_to_hash = ""
        for file in output_files:
            with open(file, "r") as f:
                string_to_hash += f.read()
        new_hash = hash(string_to_hash)
        self.logger.debug(f"Current hash set to {new_hash}")
        regenerate = old_hash is None or old_hash != new_hash
        # If different, clean directory
        if regenerate:
            self.logger.info(f"Running simulation, hash check failed")
            # Clean output dir. God I feel dangerous doing this, so hopefully unnecessary check
            if "//" not in self.output_dir and "Pippin" in self.output_dir:
                self.logger.debug(f"Cleaning output directory {self.output_dir}")
                shutil.rmtree(self.output_dir, ignore_errors=True)
                os.makedirs(self.output_dir, exist_ok=True)
            with open(hash_file, "w") as f:
                f.write(str(new_hash))
                self.logger.debug(f"New hash saved to {hash_file}")
            return False
        else:
            self.logger.info("Hash check passed, not rerunning")
            return True

    def run(self):

        regenerate = self.write_input()
        if not regenerate:
            return True

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
                sim_folder = os.path.expandvars(f"{self.global_config['SNANA']['sim_dir']}/{self.genversion}")
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

