from pippin.classifiers.classifier import Classifier
from pippin.config import mkdirs
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from pippin.task import Task
import pandas as pd
import os
from astropy.io import fits
import numpy as np


class Aggregator(Task):
    def __init__(self, name, output_dir, dependencies, options):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.passed = False
        self.classifiers = [d for d in dependencies if isinstance(d, Classifier)]
        self.output_df = os.path.join(self.output_dir, "merged.csv")
        self.id = "SNID"
        self.type_name = "SNTYPE"
        self.options = options
        self.include_type = bool(options.get("INCLUDE_TYPE", False))
        self.plot = bool(options.get("PLOT", False))

    def check_completion(self):
        return Task.FINISHED_GOOD if self.passed else Task.FINISHED_CRASH

    def check_regenerate(self):
        new_hash = self.get_hash_from_string(self.name + str(self.include_type) + str(self.plot))
        old_hash = self.get_old_hash(quiet=True)

        if new_hash != old_hash:
            self.logger.info("Hash check failed, regenerating")
            return new_hash
        else:
            self.logger.info("Hash check passed, not rerunning")
            return False

    def get_underlying_sim_tasks(self):
        tasks = []
        check = []
        for task in self.dependencies:
            for t in task.dependencies:
                check.append(t)
                if isinstance(task, SNANALightCurveFit):
                    check += task.dependencies

        for task in check:
            if isinstance(task, SNANASimulation):
                tasks.append(task)

        tasks = list(set(tasks))
        self.logger.debug(f"Found simulation dependencies: {tasks}")
        return tasks

    def run(self):
        new_hash = self.check_regenerate()
        if new_hash:
            mkdirs(self.output_dir)
            prediction_files = [d.output["predictions_filename"] for d in self.classifiers]

            df = None

            for f in prediction_files:
                dataframe = pd.read_csv(f)
                col = dataframe.columns[0]
                dataframe = dataframe.rename(columns={col: self.id})
                if df is None:
                    df = dataframe
                    self.logger.debug(f"Merging on column {self.id}")
                else:
                    df = pd.merge(df, dataframe, on=self.id, how="outer")  # Inner join atm, should I make this outer?

            if self.include_type:
                self.logger.info("Finding original types")
                sim_tasks = self.get_underlying_sim_tasks()
                type_df = None
                for s in sim_tasks:
                    phot_dir = s.output["photometry_dir"]
                    headers = [os.path.join(phot_dir, a) for a in os.listdir(phot_dir) if "HEAD" in a]
                    for h in headers:
                        with fits.open(h) as hdul:
                            data = hdul[1].data
                            snid = np.array(data.field("SNID")).astype(np.int64)
                            sntype = np.array(data.field("SNTYPE")).astype(np.int64)
                            dataframe = pd.DataFrame({self.id: snid, self.type_name: sntype})
                            if type_df is None:
                                type_df = dataframe
                            else:
                                type_df = pd.concat([type_df, dataframe])
                df = pd.merge(df, type_df, on=self.id)
            if self.plot:
                self._plot(df)

            self.logger.info(f"Merged into dataframe of {df.shape[0]} rows, with columns {df.columns}")
            df.to_csv(self.output_df, index=False, float_format="%0.4f")
            self.logger.debug(f"Saving merged dataframe to {self.output_df}")
            self.save_new_hash(new_hash)

        self.output["merge_predictions_filename"] = self.output_df
        self.output["sn_column_name"] = self.id
        if self.include_type:
            self.output["sn_type_name"] = self.type_name

        self.passed = True
        return True

    def _plot_corr(self, df):
        self.logger.debug("Making prob correlation plot")
        import matplotlib.pyplot as plt
        import seaborn as sb

        fig, ax = plt.subplots(figsize=(6, 5))
        sb.heatmap(df.corr(), ax=ax, vmin=0, vmax=1, annot=True)
        plt.show()
        if self.output_dir:
            filename = os.path.join(self.output_dir, "plt_corr.png")
            fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
            self.logger.info(f"Prob corr plot saved to {filename}")

    def _plot_prob_acc(self, df):
        self.logger.debug("Making prob accuracy plot")
        import matplotlib.pyplot as plt
        from scipy.stats import binned_statistic

        prob_bins = np.linspace(0, 1, 11)
        bin_center = 0.5 * (prob_bins[1:] + prob_bins[:-1])
        columns = [c for c in df.columns if "PROB_" in c]

        fig, ax = plt.subplots(figsize=(5, 4))
        for c in columns:
            actual_prob, _, _ = binned_statistic(df[c], df["IA"].astype(np.float), bins=prob_bins, statistic="mean")
            ax.scatter(bin_center, actual_prob, s=20, label=c)
        ax.plot(prob_bins, prob_bins, label="Expected", color="k", ls="--")
        ax.legend(loc=4, frameon=False)
        ax.set_xlabel("Reported confidence")
        ax.set_ylabel("Actual chance of being Ia")
        plt.show()
        if self.output_dir:
            filename = os.path.join(self.output_dir, "plt_prob_acc.png")
            fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
            self.logger.info(f"Prob accuracy plot saved to {filename}")

    def _plot(self, df):
        if self.type_name not in df.columns:
            self.logger.error("Cannot plot without loading in actual type. Set INCLUDE_TYPE: True in your aggregator options")
        else:
            ia = (df["SNTYPE"] == 101) | (df["SNTYPE"] == 1)
            df["IA"] = ia
            df = df.drop(["SNID", "SNTYPE"], axis=1)
            self._plot_corr(df)
            self._plot_prob_acc(df)


if __name__ == "__main__":
    df = pd.read_csv("merged.csv")
    Aggregator("AGG", "", [], {})._plot(df)