from pippin.classifiers.classifier import Classifier
from pippin.config import mkdirs
from pippin.dataprep import DataPrep
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
        self.output_df_key = os.path.join(self.output_dir, "merged.key")
        self.id = "SNID"
        self.type_name = "SNTYPE"
        self.options = options
        self.include_type = bool(options.get("INCLUDE_TYPE", False))
        self.plot = bool(options.get("PLOT", False))
        self.colours = ["#f95b4a", "#3d9fe2", "#ffa847", "#c4ef7a", "#e195e2", "#ced9ed", "#fff29b", "#903de3", "#31b58b", "#99825a"]
        self.output["classifiers"] = self.classifiers

    def _check_completion(self, squeue):
        return Task.FINISHED_SUCCESS if self.passed else Task.FINISHED_FAILURE

    def check_regenerate(self, force_refresh):
        new_hash = self.get_hash_from_string(self.name + str(self.include_type) + str(self.plot))
        old_hash = self.get_old_hash(quiet=True)

        if new_hash != old_hash:
            self.logger.info("Hash check failed, regenerating")
            return new_hash
        elif force_refresh:
            self.logger.debug("Force refresh deteted")
            return new_hash
        else:
            self.logger.info("Hash check passed, not rerunning")
            return False

    def get_underlying_sim_task(self):
        check = []
        for task in self.dependencies:
            for t in task.dependencies:
                check.append(t)
                if isinstance(task, SNANALightCurveFit):
                    check += task.dependencies

        for task in check:
            if isinstance(task, SNANASimulation) or isinstance(task, DataPrep):
                return task
        self.logger.error(f"Unable to find a simulation or data dependency for aggregator {self.name}")
        return None

    def load_prediction_file(self, filename):
        df = pd.read_csv(filename, comment="#")
        columns = df.columns
        if len(columns) == 1 and "VARNAME" in columns[0]:
            df = pd.read_csv(filename, comment="#", sep=r"\s+")
        if "VARNAMES:" in df.columns:
            df = df.drop(columns="VARNAMES:")
        remove_columns = [c for i, c in enumerate(df.columns) if i != 0 and "PROB_" not in c]
        df = df.drop(columns=remove_columns)
        return df

    def _run(self, force_refresh):
        new_hash = self.check_regenerate(force_refresh)
        if new_hash:
            mkdirs(self.output_dir)
            prediction_files = [d.output["predictions_filename"] for d in self.classifiers]
            df = None

            for f in prediction_files:
                dataframe = self.load_prediction_file(f)
                dataframe = dataframe.rename(columns={dataframe.columns[0]: self.id})
                self.logger.debug(f"Merging on column {self.id} for file {f}")
                if df is None:
                    df = dataframe
                else:
                    df = pd.merge(df, dataframe, on=self.id, how="outer")  # Inner join atm, should I make this outer?
            if self.include_type:
                self.logger.info("Finding original types")
                s = self.get_underlying_sim_task()
                type_df = None
                phot_dir = s.output["photometry_dir"]
                headers = [os.path.join(phot_dir, a) for a in os.listdir(phot_dir) if "HEAD" in a]
                if not headers:
                    self.logger.error(f"No HEAD fits files found in {phot_dir}!")
                else:
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
                        type_df.drop_duplicates(subset=self.id, inplace=True)
                df = pd.merge(df, type_df, on=self.id)

            if self.plot:
                self._plot(df)

            self.logger.info(f"Merged into dataframe of {df.shape[0]} rows, with columns {list(df.columns)}")
            df.to_csv(self.output_df, index=False, float_format="%0.4f")
            self.save_key_format(df)
            self.logger.debug(f"Saving merged dataframe to {self.output_df}")
            self.save_new_hash(new_hash)

        self.output["merge_predictions_filename"] = self.output_df
        self.output["merge_key_filename"] = self.output_df_key
        self.output["sn_column_name"] = self.id
        if self.include_type:
            self.output["sn_type_name"] = self.type_name

        self.passed = True
        return True

    def save_key_format(self, df):
        if "IA" in df.columns:
            df = df.drop(columns=[self.type_name, "IA"])
        df2 = df.fillna(0.0)
        df2.insert(0, "VARNAMES:", ["SN:"] * df2.shape[0])
        df2.to_csv(self.output_df_key, index=False, float_format="%0.4f", sep=" ")

    def _plot_corr(self, df):
        self.logger.debug("Making prob correlation plot")
        import matplotlib.pyplot as plt
        import seaborn as sb

        fig, ax = plt.subplots(figsize=(8, 6))
        df = df.dropna()
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

        prob_bins = np.linspace(0, 1, 21)
        bin_center = 0.5 * (prob_bins[1:] + prob_bins[:-1])
        columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

        fig, ax = plt.subplots(figsize=(8, 6))
        for c, col in zip(columns, self.colours):
            data, truth = self._get_data_and_truth(df[c], df["IA"])
            actual_prob, _, _ = binned_statistic(data, truth.astype(np.float), bins=prob_bins, statistic="mean")
            ax.plot(bin_center, actual_prob, label=c, c=col)
        ax.plot(prob_bins, prob_bins, label="Expected", color="k", ls="--")
        ax.legend(loc=4, frameon=False, markerfirst=False)
        ax.set_xlabel("Reported confidence")
        ax.set_ylabel("Actual chance of being Ia")
        plt.show()
        if self.output_dir:
            filename = os.path.join(self.output_dir, "plt_prob_acc.png")
            fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
            self.logger.info(f"Prob accuracy plot saved to {filename}")

    def _get_matrix(self, classified, truth):
        true_positives = classified & truth
        false_positive = classified & ~truth
        true_negative = ~classified & ~truth
        false_negative = ~classified & truth
        return true_positives.sum(), false_positive.sum(), true_negative.sum(), false_negative.sum()

    def _get_metrics(self, classified, truth):
        tp, fp, tn, fn = self._get_matrix(classified, truth)
        return {
            "purity": tp / (tp + fp),  # also known as precision
            "efficiency": tp / (tp + fn),  # also known as recall
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "specificity": fp / (fp + tn),
        }

    def _get_data_and_truth(self, data, truth, name=None):
        mask = ~(data.isna() | truth.isna())
        data = data[mask]
        truth = truth[mask].astype(np.bool)
        return data, truth

    def _plot_thresholds(self, df):
        self.logger.debug("Making threshold plot")
        import matplotlib.pyplot as plt

        thresholds = np.linspace(0.5, 0.999, 100)
        columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

        fig, ax = plt.subplots(figsize=(7, 5))
        ls = ["-", "--", ":", ":-", "-", "--", ":"]
        keys = ["purity", "efficiency"]
        for c, col in zip(columns, self.colours):
            data, truth = self._get_data_and_truth(df[c], df["IA"], name=c)
            res = {}
            for t in thresholds:
                passed = data >= t
                metrics = self._get_metrics(passed, truth)
                for key in keys:
                    if res.get(key) is None:
                        res[key] = []
                    res[key].append(metrics[key])
            for key, l in zip(keys, ls):
                ax.plot(thresholds, res[key], color=col, linestyle=l, label=f"{c[5:]} {key}")

        ax.set_xlabel("Classification probability threshold")
        ax.legend(loc=3, frameon=False, ncol=2)
        plt.show()
        if self.output_dir:
            filename = os.path.join(self.output_dir, "plt_thresholds.png")
            fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
            self.logger.info(f"Prob threshold plot saved to {filename}")

    def _plot_pr(self, df):
        self.logger.debug("Making roc plot")
        import matplotlib.pyplot as plt

        thresholds = np.linspace(0.01, 1, 100)
        columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

        fig, ax = plt.subplots(figsize=(7, 5))

        for c, col in zip(columns, self.colours):
            efficiency, purity = [], []
            data, truth = self._get_data_and_truth(df[c], df["IA"])
            for t in thresholds:
                passed = data >= t
                metrics = self._get_metrics(passed, truth)
                efficiency.append(metrics["efficiency"])
                purity.append(metrics["purity"])
            ax.plot(purity, efficiency, color=col, label=f"{c[5:]}")

        ax.set_xlabel("Precision (aka purity)")
        ax.set_xlim(0.97, 1.0)
        ax.set_ylabel("Recall (aka efficiency)")
        ax.set_title("PR Curve")
        ax.legend(frameon=False, loc=3)
        plt.show()
        if self.output_dir:
            filename = os.path.join(self.output_dir, "plt_pr.png")
            fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
            self.logger.info(f"Prob threshold plot saved to {filename}")

    def _plot_roc(self, df):
        self.logger.debug("Making pr plot")
        import matplotlib.pyplot as plt

        thresholds = np.linspace(0.01, 0.999, 100)
        columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

        fig, ax = plt.subplots(figsize=(7, 5))

        for c, col in zip(columns, self.colours):
            efficiency, specificity = [], []
            data, truth = self._get_data_and_truth(df[c], df["IA"])
            for t in thresholds:
                passed = data >= t
                metrics = self._get_metrics(passed, truth)
                efficiency.append(metrics["efficiency"])
                specificity.append(metrics["specificity"])
            ax.plot(specificity, efficiency, color=col, label=f"{c[5:]}")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(frameon=False, loc=4)
        plt.show()
        if self.output_dir:
            filename = os.path.join(self.output_dir, "plt_roc.png")
            fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
            self.logger.info(f"Prob threshold plot saved to {filename}")

    def _plot_comparison(self, df):
        self.logger.debug("Making comparison plot")
        import matplotlib.pyplot as plt

        columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

        n = len(columns)
        scale = 1.5
        fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(n * scale, n * scale))
        lim = (0, 1)
        bins = np.linspace(lim[0], lim[1], 51)

        for i, label1 in enumerate(columns):
            for j, label2 in enumerate(columns):
                ax = axes[i, j]
                if i < j:
                    ax.axis("off")
                    continue
                elif i == j:
                    h, _, _ = ax.hist(df[label1], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=self.colours[i])
                    ax.hist(df[label1], bins=bins, histtype="step", linewidth=1.5, color=self.colours[i])
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
                    ax.set_xlim(*lim)
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)
                    if j == 0:
                        ax.spines["left"].set_visible(False)
                    if j == n - 1:
                        ax.set_xlabel(label1, fontsize=6, rotation=30)
                    else:
                        ax.set_xticklabels([])
                else:
                    a1 = np.array(df[label2])
                    a2 = np.array(df[label1])
                    ax.scatter(a1, a2, s=1, c=df["SNTYPE"], cmap="Accent")
                    ax.set_xlim(*lim)
                    ax.set_ylim(*lim)
                    ax.plot(list(lim), list(lim), c="k", lw=1, alpha=0.8, ls=":")

                    if j != 0:
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                    else:
                        ax.set_ylabel(label1, fontsize=6, rotation=80)
                    if i == n - 1:
                        ax.set_xlabel(label2, fontsize=6, rotation=30)
                    else:
                        ax.set_xticklabels([])
        plt.subplots_adjust(hspace=0.0, wspace=0)
        if self.output_dir:
            filename = os.path.join(self.output_dir, "plt_scatter.png")
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True)
            self.logger.info(f"Prob scatter plot saved to {filename}")

    def _plot(self, df):
        if self.type_name not in df.columns:
            self.logger.error("Cannot plot without loading in actual type. Set INCLUDE_TYPE: True in your aggregator options")
        else:

            types = self.get_underlying_sim_task().output["types_dict"]

            ia = df["SNTYPE"].apply(lambda y: True if y in types["IA"] else (False if y in types["NONIA"] else np.nan))
            df["IA"] = ia
            df = df.drop(["SNID"], axis=1)
            self._plot_corr(df)
            self._plot_prob_acc(df)
            self._plot_thresholds(df)
            self._plot_roc(df)
            self._plot_pr(df)
            self._plot_comparison(df)

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        sim_tasks = Task.get_task_of_type(prior_tasks, SNANASimulation, DataPrep)
        classifier_tasks = Task.get_task_of_type(prior_tasks, Classifier)

        def _get_aggregator_dir(base_output_dir, stage_number, agg_name):
            return f"{base_output_dir}/{stage_number}_AGG/{agg_name}"

        tasks = []

        for agg_name in c.get("AGGREGATION", []):
            config = c["AGGREGATION"][agg_name]
            if config is None:
                config = {}
            options = config.get("OPTS", {})
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_clas = config.get("MASK_CLAS", "")
            for sim_task in sim_tasks:
                if mask_sim not in sim_task.name or mask not in sim_task.name:
                    continue
                agg_name2 = f"{agg_name}_{sim_task.name}"
                deps = [
                    c
                    for c in classifier_tasks
                    if mask in c.name and mask_clas in c.name and c.mode == Classifier.PREDICT and c.get_simulation_dependency() == sim_task
                ]
                if len(deps) == 0:
                    Task.fail_config(f"Aggregator {agg_name2} with mask {mask} matched no classifier tasks for sim {sim_task}")
                else:
                    a = Aggregator(agg_name2, _get_aggregator_dir(base_output_dir, stage_number, agg_name2), deps, options)
                    Task.logger.info(f"Creating aggregation task {agg_name2} for {sim_task.name} with {a.num_jobs} jobs")
                    tasks.append(a)
        return tasks


if __name__ == "__main__":
    df = pd.read_csv("debug/merged.csv")
    Aggregator("AGG", "debug", [], {})._plot(df)
