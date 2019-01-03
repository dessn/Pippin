import os
import inspect
import logging
import subprocess

from pippin.base import ConfigBasedExecutable


class SNANASimulation(ConfigBasedExecutable):
    def __init__(self, output_name, version, base="base_des.input"):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        self.base_file = self.data_dir + base
        super().__init__(self.base_file, output_name, ":")

        self.output_name = output_name
        self.version = version
        self.set_property("GENVERSION", version, assignment=":")
        self.config_path = self.output_dir + "/" + output_name + ".input"

    def write_input(self):
        with open(self.config_path, "w") as f:
            f.writelines(map(lambda s: s + '\n', self.base))
        self.logger.info(f"Input file written to {self.config_path}")

    def run(self):
        logging_file = self.config_path.replace(".input", ".input_log")
        with open(logging_file, "w") as f:
            subprocess.run(["snlc_sim.exe", self.config_path], stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    s = SNANASimulation("test", "testv")

    s.set_property("TESTPROP", "HELLO")
    s.delete_property("GOODBYE")
    s.write_input()

