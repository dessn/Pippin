import os

import pandas as pd
from pippin.classifiers.classifier import Classifier


class ToyClassifier(Classifier):
    def __init__(self, light_curve_dir, fit_dir, output_dir, options):
        super().__init__(light_curve_dir, fit_dir, output_dir, options)

    def classify(self):
        fitres = f"{self.fit_dir}/FITOPT000.FITRES.gz"
        self.logger.debug(f"Looking for {fitres}")
        if not os.path.exists(fitres):
            self.logger.error(f"FITRES file could not be found at {fitres}, classifer has nothing to work with")
            return False

        data = pd.read_csv(sep='\s+', skiprows=7, comment="#", compression="infer")
        print(data.columns)
        return True