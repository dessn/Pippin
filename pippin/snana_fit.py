import inspect
import os
import logging

from pippin.base import ConfigBasedExecutable


class SNANALightCurveFit(ConfigBasedExecutable):
    def __init__(self, output_name, version, base="base_des.nml", fitopts="base_des.fitopts"):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        self.base_file = self.data_dir + base
        self.fitopts_file = self.data_dir + fitopts
        super().__init__(self.base_file, output_name, "=")

        self.logger.debug(f"Loading fitopts file from {self.fitopts_file}")
        with open(self.fitopts_file, "r") as f:
            self.fitopts = list(f.read().splitlines())
            self.logger.info(f"Loaded fitopts file from {self.fitopts_file}")
        self.output_name = output_name
        self.version = version
        self.set_property("VERSION", version, assignment=":") # TODO FIX THIS, DOUBLE VERSION KEY
        self.set_property("OUTDIR",  self.output_dir, assignment=":")

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
        path = self.output_dir + "/" + self.output_name + ".nml"
        with open(path, "w") as f:
            f.writelines(map(lambda s: s + '\n', self.fitopts))
            f.writelines(map(lambda s: s + '\n', self.base))
        self.logger.info(f"NML file written to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    s = SNANALightCurveFit("test", "testv")

    s.set_snlcinp("CUTWIN_NBAND_THRESH", 1000)
    s.set_snlcinp("HELLO", "'human'")
    s.set_fitinp("FITWIN_PROB", "0.05, 1.01")
    s.set_fitinp("GOODBYE", -1)
    s.set_property("BATCH_INFO", "sbatch  $SBATCH_TEMPLATES/SBATCH_sandyb.TEMPLATE  96", assignment=":")
    s.delete_property("GOODBYE")
    s.write_nml()
