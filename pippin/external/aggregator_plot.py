import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import seaborn as sb
from scipy.stats import binned_statistic

colours = ["#f95b4a", "#3d9fe2", "#ffa847", "#c4ef7a", "#e195e2", "#ced9ed", "#fff29b", "#903de3", "#31b58b", "#99825a"]


def plot_corr(df, output_dir):
    logging.debug("Making prob correlation plot")

    fig, ax = plt.subplots(figsize=(8, 6))
    df = df.dropna()
    sb.heatmap(df.corr(), ax=ax, vmin=0, vmax=1, annot=True)
    plt.show()
    if output_dir:
        filename = os.path.join(output_dir, "plt_corr.png")
        fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
        logging.info(f"Prob corr plot saved to {filename}")


def plot_prob_acc(df, output_dir):
    logging.debug("Making prob accuracy plot")

    prob_bins = np.linspace(0, 1, 21)
    bin_center = 0.5 * (prob_bins[1:] + prob_bins[:-1])
    columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

    fig, ax = plt.subplots(figsize=(8, 6))
    for c, col in zip(columns, colours):
        data, truth = _get_data_and_truth(df[c], df["IA"])
        actual_prob, _, _ = binned_statistic(data, truth.astype(np.float), bins=prob_bins, statistic="mean")
        ax.plot(bin_center, actual_prob, label=c, c=col)
    ax.plot(prob_bins, prob_bins, label="Expected", color="k", ls="--")
    ax.legend(loc=4, frameon=False, markerfirst=False)
    ax.set_xlabel("Reported confidence")
    ax.set_ylabel("Actual chance of being Ia")
    plt.show()
    if output_dir:
        filename = os.path.join(output_dir, "plt_prob_acc.png")
        fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
        logging.info(f"Prob accuracy plot saved to {filename}")


def _get_matrix(classified, truth):
    true_positives = classified & truth
    false_positive = classified & ~truth
    true_negative = ~classified & ~truth
    false_negative = ~classified & truth
    return true_positives.sum(), false_positive.sum(), true_negative.sum(), false_negative.sum()


def _get_metrics(classified, truth):
    tp, fp, tn, fn = _get_matrix(classified, truth)
    return {
        "purity": tp / (tp + fp),  # also known as precision
        "efficiency": tp / (tp + fn),  # also known as recall
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "specificity": fp / (fp + tn),
    }


def _get_data_and_truth(data, truth):
    mask = ~(data.isna() | truth.isna())
    data = data[mask]
    truth = truth[mask].astype(np.bool)
    return data, truth


def plot_thresholds(df, output_dir):
    logging.debug("Making threshold plot")

    thresholds = np.linspace(0.5, 0.999, 100)
    columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

    fig, ax = plt.subplots(figsize=(7, 5))
    ls = ["-", "--", ":", ":-", "-", "--", ":"]
    keys = ["purity", "efficiency"]
    for c, col in zip(columns, colours):
        data, truth = _get_data_and_truth(df[c], df["IA"])
        res = {}
        for t in thresholds:
            passed = data >= t
            metrics = _get_metrics(passed, truth)
            for key in keys:
                if res.get(key) is None:
                    res[key] = []
                res[key].append(metrics[key])
        for key, l in zip(keys, ls):
            ax.plot(thresholds, res[key], color=col, linestyle=l, label=f"{c[5:]} {key}")

    ax.set_xlabel("Classification probability threshold")
    ax.legend(loc=3, frameon=False, ncol=2)
    plt.show()
    if output_dir:
        filename = os.path.join(output_dir, "plt_thresholds.png")
        fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
        logging.info(f"Prob threshold plot saved to {filename}")


def plot_pr(df, output_dir):
    logging.debug("Making roc plot")

    thresholds = np.linspace(0.01, 1, 100)
    columns = [c for c in df.columns if c.startswith("PROB_") and not c.endswith("_ERR")]

    fig, ax = plt.subplots(figsize=(7, 5))

    for c, col in zip(columns, colours):
        efficiency, purity = [], []
        data, truth = _get_data_and_truth(df[c], df["IA"])
        for t in thresholds:
            passed = data >= t
            metrics = _get_metrics(passed, truth)
            efficiency.append(metrics["efficiency"])
            purity.append(metrics["purity"])
        ax.plot(purity, efficiency, color=col, label=f"{c[5:]}")

    ax.set_xlabel("Precision (aka purity)")
    ax.set_xlim(0.97, 1.0)
    ax.set_ylabel("Recall (aka efficiency)")
    ax.set_title("PR Curve")
    ax.legend(frameon=False, loc=3)
    plt.show()
    if output_dir:
        filename = os.path.join(output_dir, "plt_pr.png")
        fig.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
        logging.info(f"Prob threshold plot saved to {filename}")


def plot_roc(self, df):
    self.logger.debug("Making pr plot")

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


def plot_comparison(df, output_dir):
    logging.debug("Making comparison plot")

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
                h, _, _ = ax.hist(df[label1], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=colours[i])
                ax.hist(df[label1], bins=bins, histtype="step", linewidth=1.5, color=colours[i])
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
                ax.set_xlim(*lim)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                if j == 0:
                    ax.spines["left"].set_visible(False)
                if j == n - 1:
                    ax.set_xlabel(label1, fontsize=6, rotation=5)
                else:
                    ax.set_xticklabels([])
            else:
                a1 = np.array(df[label2])
                a2 = np.array(df[label1])
                ax.scatter(a1, a2, s=0.5, c=df["SNTYPE"], cmap="Accent")
                ax.set_xlim(*lim)
                ax.set_ylim(*lim)
                ax.plot(list(lim), list(lim), c="k", lw=1, alpha=0.8, ls=":")

                if j != 0:
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
                else:
                    ax.set_ylabel(label1, fontsize=6, rotation=85)
                if i == n - 1:
                    ax.set_xlabel(label2, fontsize=6, rotation=5)
                else:
                    ax.set_xticklabels([])
    plt.subplots_adjust(hspace=0.0, wspace=0)
    if output_dir:
        filename = os.path.join(output_dir, "plt_scatter.png")
        fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True)
        logging.info(f"Prob scatter plot saved to {filename}")


def plot(df, output_dir):

    plot_corr(df, output_dir)
    plot_prob_acc(df, output_dir)
    plot_thresholds(df, output_dir)
    plot_roc(df, output_dir)
    plot_pr(df, output_dir)
    plot_comparison(df, output_dir)


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("mergedcsv", help="Location of merged csv to load in and plot")
    parser.add_argument("output_dir", help="Location of output_dir")
    args = parser.parse_args()

    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    logging.info(f"Input csv is {args.mergedcsv}")
    logging.info(f"Output folder {args.output_dir}")

    df = pd.read_csv(args.mergedcsv)
    plot(df, args.output_dir)
