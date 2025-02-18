import os
import pandas as pd
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs
from pippin.task import Task


class FitProbClassifier(Classifier):
    """FitProb classifier

    CONFIGURATION:
    ==============
    CLASSIFICATION:
      label:
        MASK: TEST  # partial match on sim and classifier
        MASK_SIM: TEST  # partial match on sim name
        MASK_FIT: TEST  # partial match on lcfit name
        MODE: predict
        OPTS:
            FITOPT: fitoptName # Defaults to DEFAULT, which is FITOPT000

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        prob_column_name: name of the column to get probabilities out of
        predictions_filename: location of csv filename with id/probs

    """

    def __init__(
        self,
        name,
        output_dir,
        config,
        dependencies,
        mode,
        options,
        index=0,
        model_name=None,
    ):
        super().__init__(
            name,
            output_dir,
            config,
            dependencies,
            mode,
            options,
            index=index,
            model_name=model_name,
        )
        self.output_file = None
        self.passed = False
        self.num_jobs = 1  # This is the default. Can get this from options if needed.
        self.output_file = os.path.join(self.output_dir, "predictions.csv")
        self.output["predictions_filename"] = self.output_file
        self.fitopt = options.get("FITOPT", "DEFAULT")

    def classify(self):
        new_hash = self.get_hash_from_string(self.name)
        if self._check_regenerate(new_hash):
            mkdirs(self.output_dir)
            input = self.get_fit_dependency()[0]
            if len(self.get_fit_dependency()) > 1:
                self.logger.warning(
                    f"Found more than one fit dependency for {self.name}, possibly because COMBINE_MASK is being used. FITPROB doesn't currently support this. Using first one."
                )
            fitres_file = os.path.join(
                input["fitres_dirs"][self.index], input["fitopt_map"][self.fitopt]
            )
            self.logger.debug(f"Looking for {fitres_file}")
            if not os.path.exists(fitres_file):
                self.logger.error(
                    f"FITRES file could not be found at {fitres_file}, classifer has nothing to work with"
                )
                self.passed = False
                return False

            df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")
            df = df[["CID", "FITPROB"]].rename(
                columns={"FITPROB": self.get_prob_column_name()}
            )

            self.logger.info(f"Saving probabilities to {self.output_file}")
            df.to_csv(self.output_file, index=False, float_format="%0.4f")
            chown_dir(self.output_dir)
            with open(self.done_file, "w") as f:
                f.write("SUCCESS")
            self.save_new_hash(new_hash)
        self.passed = True

        return True

    def _check_completion(self, squeue):
        if not self.passed:
            if os.path.exists(self.done_file):
                with open(self.done_file) as f:
                    self.passed = "SUCCESS" in f.read()
        return Task.FINISHED_SUCCESS if self.passed else Task.FINISHED_FAILURE

    def train(self):
        return self.classify()

    def predict(self):
        return self.classify()

    def get_unique_name(self):
        return self.name

    @staticmethod
    def get_requirements(config):
        # Does not need simulations, does light curve fits
        return False, True
