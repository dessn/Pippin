import inspect
import os
import logging
import subprocess
import time

from pippin.base import ConfigBasedExecutable


class SNANALightCurveFit(ConfigBasedExecutable):
    def __init__(self, output_dir, sim_version, config, global_config):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"

        self.config = config
        self.global_config = global_config

        base = config["BASE"]
        fitopts = config.get("FITOPTS", "empty.fitopts")

        self.base_file = self.data_dir + base
        self.fitopts_file = self.data_dir + fitopts

        super().__init__(output_dir, self.base_file, "=")

        self.logger.debug(f"Loading fitopts file from {self.fitopts_file}")
        with open(self.fitopts_file, "r") as f:
            self.fitopts = list(f.read().splitlines())
            self.logger.info(f"Loaded {len(self.fitopts)} fitopts file from {self.fitopts_file}")

        self.sim_version = sim_version
        self.config_path = self.output_dir + "/" + self.sim_version + ".nml"
        self.lc_output_dir = f"{self.output_dir}/output"

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
        # Parse config, first SNLCINP and then FITINP
        for key, value in self.config.get("SNLCINP", {}).items():
            self.set_snlcinp(key, value)
        for key, value in self.config.get("FITINP", {}).items():
            self.set_fitinp(key, value)
        self.set_property("VERSION", self.sim_version, assignment=":", section_end="&SNLCINP") # TODO FIX THIS, DOUBLE VERSION KEY
        self.set_property("OUTDIR",  self.lc_output_dir, assignment=":", section_end="&SNLCINP")

        # Write main file
        with open(self.config_path, "w") as f:
            f.writelines(map(lambda s: s + '\n', self.fitopts))
            f.writelines(map(lambda s: s + '\n', self.base))
        self.logger.info(f"NML file written to {self.config_path}")
        assert os.path.exists(self.config_path), "NML file does not exist"

    def run(self):
        self.write_nml()
        logging_file = self.config_path.replace(".nml", ".nml_log")
        with open(logging_file, "w") as f:
            # TODO: Add queue to config and run
            subprocess.run(["split_and_fit.pl", os.path.basename(self.config_path), "NOPROMPT"], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
        self.logger.info(f"Light curve fitting outputting to {logging_file}")
        done_file = f"{self.output_dir}/SPLIT_JOBS_LCFIT.tar.gz"
        secondary_log = f"{self.output_dir}/SPLIT_JOBS_LCFIT/MERGELOGS/MERGE2.LOG"

        log_files = [logging_file, secondary_log]
        while True:
            time.sleep(self.global_config["OUTPUT"].getint("ping_frequency"))

            # Check for errors
            for file in log_files:
                if os.path.exists(file):
                    with open(file, "r") as f:
                        output_error = False
                        for line in f.read().splitlines():
                            if "ERROR" in line and not output_error:
                                self.logger.critical(f"Fatal error in light curve fitting. See {file} for details.")
                                output_error = True
                            if output_error:
                                self.logger.error(f"Excerpt: {line}")

                    if output_error:
                        return False

            # Check for existence of SPLIT_JOBS_LCFIT.tar.gz to see if job is done
            if os.path.exists(done_file):
                self.logger.info("Tarball found, fitting successful")
                return True


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
