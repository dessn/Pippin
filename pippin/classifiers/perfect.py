import os
import shutil
from collections import defaultdict

import pandas as pd
from astropy.io import fits
import numpy as np
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs
from pippin.task import Task


class PerfectClassifier(Classifier):
    """Classification task for the SuperNNova classifier.

    CONFIGURATION
    =============

    CLASSIFICATION:
        label:
            MASK_SIM: mask  # partial match
            MASK_FIT: mask  # partial match
            MASK: mask  # partial match
            MODE: predict
            OPTS:
                PROB_IA: 1.0 # Sets IA prob to this number Default to 1.
                PROB_CC: 0.01 # Sets CC prob to this number. Default to 0. Used for all NONIA - so for data without known type, having PROB_CC = 0 effectively removes untyped

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
        self.prob_ia = options.get("PROB_IA", 1.0)
        self.prob_cc = options.get("PROB_CC", 0.0)
        self.output_file = os.path.join(self.output_dir, "predictions.csv")
        self.output.update({"predictions_filename": self.output_file})

    def get_unique_name(self):
        return self.name

    def classify(self):
        new_hash = self.get_hash_from_string(
            self.name + f"{self.prob_ia}_{self.prob_cc}"
        )

        if self._check_regenerate(new_hash):
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            try:
                name = self.get_prob_column_name()
                cid = "CID"
                s = self.get_simulation_dependency()
                df = None
                phot_dirs = [
                    sim_dep.output["photometry_dirs"][self.index] for sim_dep in s
                ]
                headers = [
                    os.path.join(phot_dir, a)
                    for phot_dir in phot_dirs
                    for a in os.listdir(phot_dir)
                    if "HEAD" in a
                ]
                if not headers:
                    Task.fail_config(f"No HEAD fits files found in {phot_dir}!")
                else:
                    types = defaultdict(list)
                    for t in self.get_simulation_dependency():
                        for k, v in t.output["types_dict"].items():
                            types[k] = np.unique(types[k] + v)
                    self.logger.debug(f"Input types are {types}")

                    for h in headers:
                        with fits.open(h) as hdul:
                            data = hdul[1].data
                            snid = np.array(data.field("SNID"))
                            sntype = np.array(data.field("SNTYPE")).astype(np.int64)
                            is_ia = np.isin(sntype, types["IA"])
                            prob = (is_ia * self.prob_ia) + (~is_ia * self.prob_cc)

                            dataframe = pd.DataFrame({cid: snid, name: prob})
                            dataframe[cid] = dataframe[cid].apply(str)
                            dataframe[cid] = dataframe[cid].str.strip()
                            if df is None:
                                df = dataframe
                            else:
                                df = pd.concat([df, dataframe])
                    df.drop_duplicates(subset=cid, inplace=True)

                self.logger.info(f"Saving probabilities to {self.output_file}")
                df.to_csv(self.output_file, index=False, float_format="%0.4f")
                chown_dir(self.output_dir)
                with open(self.done_file, "w") as f:
                    f.write("SUCCESS")
                self.save_new_hash(new_hash)
            except Exception as e:
                self.logger.exception(e, exc_info=True)
                self.passed = False
                with open(self.done_file, "w") as f:
                    f.write("FAILED")
                return False
        else:
            self.should_be_done()
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

    @staticmethod
    def get_requirements(config):
        return True, False
