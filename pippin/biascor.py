import inspect
import shutil
import subprocess
import os

from pippin.base import ConfigBasedExecutable
from pippin.config import chown_dir, mkdirs, get_config
from pippin.task import Task


class BiasCor(ConfigBasedExecutable):
    def __init__(self, name, output_dir, dependencies, options, config):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        super().__init__(name, output_dir, os.path.join(self.data_dir, "bbc.input"), "=", dependencies=dependencies)

        self.options = options
        self.config = config
        self.logging_file = os.path.join(self.output_dir, "output.log")
        self.global_config = get_config()

        self.merged_data = config.get("DATA")
        self.merged_iasim = config.get("SIMFILE_BIASCOR")
        self.merged_ccsim = config.get("SIMFILE_CCPRIOR")
        self.classifier = config.get("CLASSIFIER")

        self.bias_cor_fits = None
        self.cc_prior_fits = None
        self.data = None
        self.sim_names = [m.output["sim_name"] for m in self.merged_data]
        self.genversion = "_".join(self.sim_names) + "_" + self.classifier.name

        self.config_filename = f"{self.genversion}.input"  # Make sure this syncs with the tmp file name
        self.config_path = os.path.join(self.output_dir, self.config_filename)
        self.fit_output_dir = os.path.join(self.output_dir, "output")
        self.done_file = os.path.join(self.fit_output_dir, f"FITJOBS/ALL.DONE")
        self.probability_column_name = self.classifier.output["prob_column_name"]

        self.output["fit_output_dir"] = self.fit_output_dir

        # calculate genversion the hard way
        print(self.merged_data[0].output)
        a_genversion = self.merged_data[0].output["genversion"]
        for n in self.sim_names:
            a_genversion = a_genversion.replace(n, "")
        while a_genversion.endswith("_"):
            a_genversion = a_genversion[:-1]
        self.output["subdir"] = a_genversion
        self.output["muopts"] = (self.config.get("MUOPTS", {}).keys())

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug("Done file found, biascor task finishing")
            with open(self.done_file) as f:
                if "FAIL" in f.read():
                    self.logger.error("Done file reporting failure!")
                    return Task.FINISHED_FAILURE
            return Task.FINISHED_SUCCESS
        if os.path.exists(self.logging_file):
            with open(self.logging_file) as f:
                output_error = False
                for line in f.read().splitlines():
                    if "ABORT ON FATAL ERROR" in line:
                        self.logger.error(f"Output log showing abort: {self.logging_file}")
                        output_error = True
                    if output_error:
                        self.logger.error(line)
                if output_error:
                    return Task.FINISHED_FAILURE
        return 1

    def write_input(self, force_refresh):
        self.bias_cor_fits = ",".join([m.output["fitres_file"] for m in self.merged_iasim])
        self.cc_prior_fits = None if self.merged_ccsim is None else ",".join([m.output["fitres_file"] for m in self.merged_ccsim])
        self.data = [m.output["output_dir"] for m in self.merged_data]

        self.set_property("simfile_biascor", self.bias_cor_fits)
        self.set_property("simfile_ccprior", self.cc_prior_fits)
        self.set_property("varname_pIa", self.probability_column_name)
        self.set_property("OUTDIR_OVERRIDE", self.fit_output_dir, assignment=": ")
        self.set_property("STRINGMATCH_IGNORE", " ".join(self.sim_names), assignment=": ")
        if self.options.get("BATCH_INFO"):
            self.set_property("BATCH_INFO", self.options.get("BATCH_INFO"), assignment=": ")

        # self.set_property("INPDIR", ",".join(self.data))
        bullshit_hack = ""
        for i, d in enumerate(self.data):
            if i > 0:
                bullshit_hack += "\nINPDIR+: "
            bullshit_hack += d
        self.set_property("INPDIR+", bullshit_hack, assignment=": ")

        # Set MUOPTS at top of file
        mu_str = ""
        for label, value in self.config.get("MUOPTS").items():
            if mu_str != "":
                mu_str += "\nMUOPT: "
            mu_str += f"[{label}] "
            if value.get("SIMFILE_BIASCOR"):
                mu_str += f"simfile_biascor={','.join([v.output['fitres_file'] for v in value.get('SIMFILE_BIASCOR')])} "
            if value.get("SIMFILE_CCPRIOR"):
                mu_str += f"simfile_ccprior={','.join([v.output['fitres_file'] for v in value.get('SIMFILE_CCPRIOR')])} "
            if value.get("CLASSIFIER"):
                mu_str += f"varname_pIa={value.get('CLASSIFIER').output['prob_column_name']} "
            if value.get("FITOPT") is not None:
                mu_str += f"FITOPT={value.get('FITOPT')} "
            mu_str += "\n"
        self.set_property("MUOPT", mu_str, assignment=": ", section_end="#MUOPT_END")

        final_output = "\n".join(self.base)

        new_hash = self.get_hash_from_string(final_output)
        old_hash = self.get_old_hash()

        if force_refresh or new_hash != old_hash:
            self.logger.debug("Regenerating results")

            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)

            with open(self.config_path, "w") as f:
                f.writelines(final_output)
            self.logger.info(f"Input file written to {self.config_path}")

            self.save_new_hash(new_hash)
            return True
        else:
            self.logger.debug("Hash check passed, not rerunning")
            return False

    def _run(self, force_refresh):
        regenerating = self.write_input(force_refresh)
        if regenerating:
            command = ["SALT2mu_fit.pl", self.config_filename, "NOPROMPT"]
            self.logger.debug(f"Will check for done file at {self.done_file}")
            self.logger.debug(f"Will output log at {self.logging_file}")
            self.logger.debug(f"Running command: {' '.join(command)}")
            with open(self.logging_file, "w") as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir)
            chown_dir(self.output_dir)
        return True
