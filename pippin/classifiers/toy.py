import os

import pandas as pd
import numpy as np
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs
from pippin.task import Task


class ToyClassifier(Classifier):

    def __init__(self, name, light_curve_dir, fit_dir, output_dir, mode, options):
        super().__init__(name, light_curve_dir, fit_dir, output_dir,  mode, options)
        self.output_file = None
        self.passed = None

    def classify(self):
        mkdirs(self.output_dir)

        fitres = f"{self.fit_dir}/FITOPT000.FITRES.gz"
        self.logger.debug(f"Looking for {fitres}")
        if not os.path.exists(fitres):
            self.logger.error(f"FITRES file could not be found at {fitres}, classifer has nothing to work with")
            self.passed = False
            return False

        data = pd.read_csv(fitres, sep='\s+', comment="#", compression="infer")
        ids = data["CID"].values
        probability = np.random.uniform(size=ids.size)
        combined = np.vstack((ids, probability)).T

        self.output_file = self.output_dir + "/prob.txt"
        self.logger.info(f"Saving probabilities to {self.output_file}")
        np.savetxt(self.output_file, combined)
        chown_dir(self.output_dir)
        self.passed = True
        return True

    def check_completion(self):
        self.output = {"predictions": self.output_file}
        return Task.FINISHED_GOOD if self.passed else Task.FINISHED_CRASH

    def train(self):
        return self.classify()

    def predict(self):
        return self.classify()
