import os
import shutil
import subprocess

import pandas as pd
from astropy.io import fits
import numpy as np
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs
from pippin.task import Task


class UnityClassifier(Classifier):
    """ Classification task for the SuperNNova classifier.

    CONFIGURATION
    =============

    CLASSIFICATION:
        label:
            MASK_SIM: mask  # partial match
            MASK_FIT: mask  # partial match
            MASK: mask  # partial match
            MODE: predict

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        prob_column_name: name of the column to get probabilities out of
        predictions_filename: location of csv filename with id/probs


    """

    def __init__(self, name, output_dir, dependencies, mode, options, index=0, model_name=None):
        super().__init__(name, output_dir, dependencies, mode, options, index=index, model_name=model_name)
        self.output_file = None
        self.passed = False
        self.num_jobs = 1  # This is the default. Can get this from options if needed.
        self.output_file = os.path.join(self.output_dir, "predictions.csv")
        self.output.update({"predictions_filename": self.output_file})

    def get_unique_name(self):
        return "UNITY"

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
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            try:
                name = self.get_prob_column_name()
                cid = "CID"
                s = self.get_simulation_dependency()
                df = None
                phot_dir = s.output["photometry_dirs"][self.index]
                headers = [os.path.join(phot_dir, a) for a in os.listdir(phot_dir) if "HEAD" in a]
                if len(headers) == 0:
                    self.logger.warning(f"No HEAD fits files found in {phot_dir}! Going to do it manually, this may not work.")

                    cmd = "grep --exclude-dir=* SNID: * | awk -F ':' '{print $3}'"
                    self.logger.debug(f"Running command   {cmd}")
                    process = subprocess.run(cmd, capture_output=True, cwd=phot_dir, shell=True)
                    output = process.stdout.decode("ascii").split("\n")

                    snid = [x.strip() for x in output]
                    df = pd.DataFrame({cid: snid, name: np.ones(len(snid))})
                    df.drop_duplicates(subset=cid, inplace=True)

                else:
                    for h in headers:
                        with fits.open(h) as hdul:
                            data = hdul[1].data
                            snid = np.array(data.field("SNID"))
                            dataframe = pd.DataFrame({cid: snid, name: np.ones(snid.shape)})
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
        return Task.FINISHED_SUCCESS if self.passed else Task.FINISHED_FAILURE

    def train(self, force_refresh):
        return self.classify(force_refresh)

    def predict(self, force_refresh):
        return self.classify(force_refresh)

    @staticmethod
    def get_requirements(config):
        return True, False
