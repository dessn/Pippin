import inspect
import os
import logging
import subprocess
import time
import pandas as pd

from pippin.base import ConfigBasedExecutable
from pippin.config import get_hash, chown_dir
from pippin.task import Task


class SNANALightCurveFit(ConfigBasedExecutable):
    def __init__(self, name, output_dir, sim_version, config, global_config):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"

        self.config = config
        self.global_config = global_config

        base = config["BASE"]
        fitopts = config.get("FITOPTS", "empty.fitopts")
        self.base_file = self.data_dir + base
        self.fitopts_file = self.data_dir + fitopts

        super().__init__(name, output_dir, self.base_file, "=")

        self.sim_version = sim_version
        self.config_path = self.output_dir + "/" + self.sim_version + ".nml"
        self.lc_output_dir = f"{self.output_dir}/output"
        self.set_num_jobs(int(config.get("NUM_JOBS", 100)))

        self.logging_file = self.config_path.replace(".nml", ".nml_log")
        self.done_file = f"{self.lc_output_dir}/SPLIT_JOBS_LCFIT.tar.gz"
        secondary_log = f"{self.lc_output_dir}/SPLIT_JOBS_LCFIT/MERGELOGS/MERGE2.LOG"

        self.log_files = [self.logging_file, secondary_log]

    def print_stats(self):
        folders = [f for f in os.listdir(self.lc_output_dir) if f.startswith("PIP_") and os.path.isdir(self.lc_output_dir + "/" + f)]
        for f in folders:
            path = os.path.join(self.lc_output_dir, f)
            data = pd.read_csv(os.path.join(path, "FITOPT000.FITRES.gz"), sep='\s+', comment="#", compression="infer")
            counts = data.groupby("TYPE").size()
            self.logger.info("Types:  " + ("  ".join([f"{k}:{v}" for k, v in zip(counts.index, counts.values)])))

    def set_snlcinp(self, name, value):
        """ Ensures the property name value pair is set in the SNLCINP section.

        Parameters
        ----------
        name : str
            The name of the property. Case insensitive, will be cast to upper.
        value : object
            The value to use. Object will be cast to string. For strings, include single quotes.
        """
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
        self.set_property(name, value, section_start="&FITINP", section_end="&END")

    def write_nml(self):
        self.logger.debug(f"Loading fitopts file from {self.fitopts_file}")
        with open(self.fitopts_file, "r") as f:
            self.fitopts = list(f.read().splitlines())
            self.logger.info(f"Loaded {len(self.fitopts)} fitopts file from {self.fitopts_file}")

        # Parse config, first SNLCINP and then FITINP
        for key, value in self.config.get("SNLCINP", {}).items():
            self.set_snlcinp(key, value)
        for key, value in self.config.get("FITINP", {}).items():
            self.set_fitinp(key, value)
        self.set_property("VERSION", self.sim_version + "*", assignment=":", section_end="&SNLCINP") # TODO FIX THIS, DOUBLE VERSION KEY
        self.set_property("OUTDIR",  self.lc_output_dir, assignment=":", section_end="&SNLCINP")

        # We want to do our hashing check here
        string_to_hash = self.fitopts + self.base
        with open(os.path.abspath(inspect.stack()[0][1]), "r") as f:
            string_to_hash += f.read()
        new_hash = self.get_hash_from_string("".join(string_to_hash))
        old_hash = self.get_old_hash()
        regenerate = old_hash is None or old_hash != new_hash

        if regenerate:
            self.logger.info(f"Running Light curve fit, hash check failed")

            # Write main file
            with open(self.config_path, "w") as f:
                f.writelines(map(lambda s: s + '\n', string_to_hash))
            self.logger.info(f"NML file written to {self.config_path}")
            self.save_new_hash(new_hash)
            chown_dir(self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")

        return regenerate, new_hash

    def run(self):
        regenerate, new_hash = self.write_nml()
        if not regenerate:
            return new_hash
        self.logger.info(f"Light curve fitting outputting to {self.logging_file}")
        with open(self.logging_file, "w") as f:
            # TODO: Add queue to config and run
            subprocess.run(["split_and_fit.pl", self.config_path, "NOPROMPT"], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)

    def check_completion(self):
        # Check for errors
        for file in self.log_files:
            if os.path.exists(file):
                with open(file, "r") as f:
                    output_error = False
                    for line in f.read().splitlines():
                        if ("ERROR" in line or ("ABORT" in line and " 0 " not in line)) and not output_error:
                            self.logger.critical(f"Fatal error in light curve fitting. See {file} for details.")
                            output_error = True
                        if output_error:
                            self.logger.error(f"Excerpt: {line}")

                if output_error:
                    return Task.FINISHED_CRASH

        # Check for existence of SPLIT_JOBS_LCFIT.tar.gz to see if job is done
        if os.path.exists(self.done_file):
            self.logger.info("Tarball found, fitting complete, cleaning up the directory")
            try:
                logging_file2 = self.logging_file.replace("_log", "_log2")
                with open(logging_file2, "w") as f:
                    subprocess.run(["split_and_fit.pl", "CLEANMASK", "4", "NOPROMPT"], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir, check=True)
                    time.sleep(10)
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"split_and_fit.pl has a return code of {e.returncode}. This may or may not be an issue.")
            chown_dir(self.output_dir)
            self.print_stats()
            return Task.FINISHED_GOOD
        return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    config = {"BASE": "des.nml", }
    s = SNANALightCurveFit("../output/test", "testv", config, {})

    s.set_snlcinp("CUTWIN_NBAND_THRESH", 1000)
    s.set_snlcinp("HELLO", "'human'")
    s.set_fitinp("FITWIN_PROB", "0.05, 1.01")
    s.set_fitinp("GOODBYE", -1)
    s.set_property("BATCH_INFO", "sbatch  $SBATCH_TEMPLATES/SBATCH_sandyb.TEMPLATE  96", assignment=":")
    s.delete_property("GOODBYE")
    s.write_nml()
