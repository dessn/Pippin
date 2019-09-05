import os
import pandas as pd
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs
from pippin.task import Task


class FitProbClassifier(Classifier):

    def __init__(self, name, output_dir, dependencies, mode, options):
        super().__init__(name, output_dir, dependencies,  mode, options)
        self.output_file = None
        self.passed = False
        self.num_jobs = 1  # This is the default. Can get this from options if needed.
        self.output_file = os.path.join(self.output_dir, "predictions.csv")

    def check_regenerate(self, force_refresh):

        new_hash = self.get_hash_from_string(self.name)
        old_hash = self.get_old_hash(quiet=True)

        if new_hash != old_hash:
            self.logger.info("Hash check failed, regenerating")
            return new_hash
        elif force_refresh:
            self.logger.debug("Force refresh, regenerating")
            return new_hash
        else:
            self.logger.info("Hash check passed, not rerunning")
            return False

    def classify(self, force_refresh):
        new_hash = self.check_regenerate(force_refresh)
        if new_hash:
            mkdirs(self.output_dir)
            input = self.get_fit_dependency()
            fitres_file = input["fitres_file"]
            self.logger.debug(f"Looking for {fitres_file}")
            if not os.path.exists(fitres_file):
                self.logger.error(f"FITRES file could not be found at {fitres_file}, classifer has nothing to work with")
                self.passed = False
                return False

            df = pd.read_csv(fitres_file, sep='\s+', comment="#", compression="infer")
            df = df[["CID", "FITPROB"]].rename(columns={"FITPROB": self.get_prob_column_name()})

            self.logger.info(f"Saving probabilities to {self.output_file}")
            df.to_csv(self.output_file, index=False, float_format="%0.4f")
            chown_dir(self.output_dir)
            with open(self.done_file, "w") as f:
                f.write("SUCCESS")
            self.save_new_hash(new_hash)
        self.passed = True

        return True

    def _check_completion(self, squeue):
        self.output.update({
            "predictions_filename": self.output_file
        })
        return Task.FINISHED_SUCCESS if self.passed else Task.FINISHED_FAILURE

    def train(self, force_refresh):
        return self.classify(force_refresh)

    def predict(self, force_refresh):
        return self.classify(force_refresh)

    @staticmethod
    def get_requirements(config):
        # Does not need simulations, does light curve fits
        return False, True