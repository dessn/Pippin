import inspect
import shutil
import subprocess

from pippin.aggregator import Aggregator
from pippin.base import ConfigBasedExecutable
from pippin.config import chown_dir, mkdirs
from pippin.snana_fit import SNANALightCurveFit
from pippin.task import Task
import os


class BiasCor(ConfigBasedExecutable):
    def __init__(self, name, output_dir, dependencies, options, classifier_task):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        super().__init__(name, output_dir, os.path.join(self.data_dir, "bbc.input"), "=", dependencies=dependencies)

        self.options = options
        self.logfile = os.path.join(self.output_dir, "output.log")

        self.bias_cor_fits = self.options.get("BIAS_COR_FITS")  # merge task(s) or fitres file
        self.cc_prior_fits = self.options.get("CC_PRIOR_FITS")  # merge task(s) or fitres file

        if self.bias_cor_fits is None:
            self.logger.error("Please set the BIAS_COR_FITS option to the merge task for the Ia only classified FITRES file (or point to the fitres file)")
            raise ValueError("BIAS_COR_FITS not found")
        if self.cc_prior_fits is None:
            self.logger.error("Please set the CC_PRIOR_FITS option to the merge task for the CC only classified FITRES file (or point to the fitres file)")
            raise ValueError("CC_PRIOR_FITS not found")

        if isinstance(self.bias_cor_fits, list):
            self.bias_cor_fits = ",".join(self.bias_cor_fits)
        if isinstance(self.cc_prior_fits, list):
            self.cc_prior_fits = ",".join(self.cc_prior_fits)

        self.classifier_task = classifier_task
        self.probability_column_name = classifier_task.output["prob_column_name"]

    def _check_completion(self, squeue):
        pass

    def _run(self, force_refresh):
        pass
