import os
import inspect
import logging
import shutil
import subprocess

from pippin.base import ConfigBasedExecutable


class SNANASimulation(ConfigBasedExecutable):
    def __init__(self, output_dir, genversion, config, combine="combine.input"):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"

        super().__init__(output_dir, self.data_dir + combine, ":")

        self.genversion = genversion
        self.set_property("GENVERSION", genversion, assignment=":", section_end="ENDLIST_GENVERSION")
        self.config_path = self.output_dir + "/" + genversion + ".input"
        self.base_ia = config["IA"]["BASE"]
        self.base_cc = config["NONIA"]["BASE"]

        shutil.copy(self.data_dir + self.base_ia, self.output_dir)
        shutil.copy(self.data_dir + self.base_cc, self.output_dir)

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

    def write_input(self):
        with open(self.config_path, "w") as f:
            f.writelines(map(lambda s: s + '\n', self.base))
        self.logger.info(f"Input file written to {self.config_path}")

    def run(self):
        self.write_input()
        logging_file = self.config_path.replace(".input", ".input_log")
        with open(logging_file, "w") as f:
            subprocess.run(["snlc_SNmix.pl", self.config_path], stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)7s |%(funcName)20s]   %(message)s")
    s = SNANASimulation("test", "testv")

    s.set_property("TESTPROP", "HELLO")
    s.delete_property("GOODBYE")
    s.write_input()

