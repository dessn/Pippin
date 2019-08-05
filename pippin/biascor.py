import inspect
import shutil
import subprocess
import os

from pippin.base import ConfigBasedExecutable
from pippin.config import chown_dir, mkdirs, get_config
from pippin.task import Task


class BiasCor(ConfigBasedExecutable):
    def __init__(self, name, output_dir, dependencies, options, merged_data, merged_iasim, merged_ccsim, classifier):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        super().__init__(name, output_dir, os.path.join(self.data_dir, "bbc.input"), "=", dependencies=dependencies)

        self.options = options
        self.logging_file = os.path.join(self.output_dir, "output.log")
        self.global_config = get_config()

        self.merged_data = merged_data
        self.merged_iasim = merged_iasim
        self.merged_ccsim = merged_ccsim

        self.bias_cor_fits = None
        self.cc_prior_fits = None
        self.data = None
        self.genversion = "_".join([m.get_lcfit_dep()["sim_name"] for m in merged_data]) + "_" + classifier.name

        self.config_filename = f"{self.genversion}.input"  # Make sure this syncs with the tmp file name
        self.config_path = os.path.join(self.output_dir, self.config_filename)
        self.fit_output_dir = os.path.join(self.output_dir, "output")
        self.done_file = os.path.join(self.fit_output_dir, f"FITJOBS/ALL.DONE")
        self.probability_column_name = classifier.output["prob_column_name"]

        self.output["fit_output_dir"] = self.fit_output_dir

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

        # self.set_property("INPDIR", ",".join(self.data))
        self.set_property("INPDIR", self.data[0])
        if len(self.data) > 1:
            self.set_property("INPDIR+", self.data[1])

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
