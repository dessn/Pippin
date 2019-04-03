import os

import pandas as pd
import numpy as np
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs


class NearestNeighborClassifier(Classifier):
    def __init__(self, light_curve_dir, fit_dir, output_dir, options):
        super().__init__(light_curve_dir, fit_dir, output_dir, options)
        self.logger.info(f"Creating Nearest Neighbor classifier with options: {options}")

    def classify(self):
        mkdirs(self.output_dir)

        fitres = f"{self.fit_dir}/FITOPT000.FITRES.gz"
        self.logger.debug(f"Looking for {fitres}")
        if not os.path.exists(fitres):
            self.logger.error(f"FITRES file could not be found at {fitres}, classifer has nothing to work with")
            return False

        data = pd.read_csv(fitres, sep='\s+', comment="#", compression="infer")
        ids = data["CID"].values
        probability = np.random.uniform(size=ids.size)
        combined = np.vstack((ids, probability)).T

        output_file = self.output_dir + "/prob.txt"
        self.logger.info(f"Saving probabilities to {output_file}")
        np.savetxt(output_file, combined)
        chown_dir(self.output_dir)
        return True