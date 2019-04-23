from pippin.classifiers.classifier import Classifier
from pippin.config import mkdirs
from pippin.task import Task
import pandas as pd
import os


class Aggregator(Task):
    def __init__(self, name, output_dir, dependencies):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.passed = False
        self.classifiers = [d for d in dependencies if isinstance(d, Classifier)]
        self.output_df = os.path.join(self.output_dir, "merged.csv")

    def check_completion(self):
        return Task.FINISHED_GOOD if self.passed else Task.FINISHED_CRASH

    def check_regenerate(self):

        new_hash = self.get_hash_from_string(self.name)
        old_hash = self.get_old_hash(quiet=True)

        if new_hash != old_hash:
            self.logger.info("Hash check failed, regenerating")
            return new_hash
        else:
            self.logger.info("Hash check passed, not rerunning")
            return False

    def run(self):
        new_hash = self.check_regenerate()
        if new_hash:
            mkdirs(self.output_dirt)
            prediction_files = [d.output["predictions_filename"] for d in self.classifiers]

            df = None
            col = "CID"
            for f in prediction_files:
                dataframe = pd.read_csv(f)
                if df is None:
                    df = dataframe
                    col = dataframe.columns[0]
                    self.logger.debug(f"Merging on column {col}")
                else:
                    df = pd.merge(df, dataframe, on=col)  # Inner join atm, should I make this outer?

            self.logger.info(f"Merged into dataframe of {df.shape[0]} rows, with columns {df.columns}")
            df.to_csv(self.output_df, index=False, float_format="%0.4f")
            self.logger.debug(f"Saving merged dataframe to {self.output_df}")
            self.save_new_hash(new_hash)

        self.output["merge_predictions_filename"] = self.output_df
        self.passed = True
        return True
