import numpy as np
import yaml
from chainconsumer import ChainConsumer
import pandas as pd
import sys
import argparse
import os
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def setup_logging():
    fmt = "[%(levelname)8s |%(funcName)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler("plot_biascor.log")])
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("chainconsumer").setLevel(logging.WARNING)


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input yml file", type=str)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config.update(config["BIASCOR"])

    return config


def load_file(file, args):
    np.random.seed(123)
    logging.info(f"Loading in file {file}")
    if not os.path.exists(file):
        logging.warning(f"File {file} not count, continuing and hoping this is executing on your local compute and you have the summary")
        return None
    df = pd.read_csv(file)
    if "w" in args.get("BLIND", []):
        df["w"] += np.random.normal(loc=0, scale=0.2, size=1000)[343]
    if "om" in args.get("BLIND", []):
        df["OM"] += np.random.normal(loc=0, scale=0.1, size=1000)[432]
    return df


def plot_single_file(source_file, df):
    logging.info(f"Plotting single file {source_file}")
    name = os.path.basename(os.path.dirname(os.path.dirname(source_file)))
    output_file = name + ".png"
    logging.info(f"Creating wfit plot output to {output_file}")

    c = ChainConsumer()
    labels = [r"$\Omega_m$", "$w$", r"$\sigma_{int}$"]
    for index, row in df.iterrows():
        means = [row["OM"], row["w"], row["sigint"]]
        cov = np.diag([row["OM_sig"] ** 2, row["wsig_marg"] ** 2, 0.01 ** 2])
        c.add_covariance(means, cov, parameters=labels, name=f"Realisation {index}")
    c.plotter.plot_summary(errorbar=True, filename=output_file)
    del c


def make_summary_file(wfit_files):
    logging.info("Creating summary file")
    all_output_csv = "all_biascor_individual.csv"

    if os.path.exists(all_output_csv):
        df_all = pd.read_csv(all_output_csv)
    else:
        df_all = None
        for f in wfit_files:
            df = pd.read_csv(f)
            name = os.path.basename(os.path.dirname(os.path.dirname(f)))
            df["name"] = name
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])
        df_all.to_csv(all_output_csv, index=False, float_format="%0.5f")
    return df_all


def plot_all_files(df_all):
    logging.info("Plotting all files")
    output_file = "all_biascor.png"

    c = ChainConsumer()
    labels = [r"$\Omega_m$", "$w$", r"$\sigma_{int}$"]
    data = []
    for name, df in df_all.groupby("name"):
        means = [df["OM"].mean(), df["w"].mean(), df["sigint"].mean()]
        if df.shape[0] < 2:
            name2 = name + " (showing mean error)"
            cov = np.diag([df["OM_sig"].mean() ** 2, df["wsig_marg"].mean() ** 2, 0.01 ** 2])
        else:
            name2 = name + " (showing scatter error)"
            cov = np.diag([df["OM"].std() ** 2, df["w"].std() ** 2, df["sigint"].std() ** 2])
        c.add_covariance(means, cov, parameters=labels, name=name2.replace("_", "\\_"))
        data.append([name, df["w"].mean(), df["w"].std(), df["wsig_marg"].mean()])
    wdf = pd.DataFrame(data, columns=["name", "mean_w", "scatter_mean_w", "mean_std_w"])
    wdf.to_csv(output_file.replace(".png", ".csv"), index=False, float_format="%0.4f")
    c.plotter.plot_summary(errorbar=True, filename=output_file)


def plot_scatter_comp(df_all):
    logging.info("Creating scatter plots")
    cols = ChainConsumer()._all_colours * 2
    # Cant plot data, want to make sure all the versions match
    # So split these into groups base on how many versions
    res = {}
    for name, df in df_all.groupby("name"):
        if df.shape[0] > 1:
            key = df.shape[0]
            if res.get(key) is None:
                res[key] = []
            res[key].append((name, df["w"].values))
    for key, value in res.items():
        if key < 2:
            continue
        logging.info(f"Creating scatter plot for key {key}")
        n = len(value)
        labels = [v[0].replace("_", "\n") for v in value]
        ws = np.array([v[1] for v in value])
        num_bins = 1 + int(1.5 * np.ceil(np.sqrt(key)))
        min_w = ws.min()
        max_w = ws.max()
        bins = np.linspace(min_w, max_w, num_bins)
        lim = (min_w - 0.001, max_w + 0.001)

        fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(2 * n, 2 * n), sharex=True)
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                ax = axes[i, j]
                if i < j:
                    ax.axis("off")
                    continue
                elif i == j:
                    h, _, _ = ax.hist(ws[i, :], bins=bins, histtype="stepfilled", linewidth=2, alpha=0.3, color=cols[i])
                    ax.hist(ws[i, :], bins=bins, histtype="step", linewidth=1.5, color=cols[i])
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
                    ax.set_xlim(*lim)
                    if bins[0] < -1 < bins[-1]:
                        yval = interp1d(0.5 * (bins[:-1] + bins[1:]), h, kind="nearest")([-1.0])[0]
                        ax.plot([-1.0, -1.0], [0, yval], color="k", lw=1, ls="--", alpha=0.4)
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)
                    if j == 0:
                        ax.spines["left"].set_visible(False)
                    if j == n - 1:
                        ax.set_xlabel(label2, fontsize=10)
                else:
                    a1 = ws[j, :]
                    a2 = ws[i, :]
                    c = np.abs(a1 - a2)
                    ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.02, vmax=0.05)
                    ax.set_xlim(*lim)
                    ax.set_ylim(*lim)
                    ax.plot([min_w, max_w], [min_w, max_w], c="k", lw=1, alpha=0.8, ls=":")
                    ax.axvline(-1.0, color="k", lw=1, ls="--", alpha=0.4)
                    ax.axhline(-1.0, color="k", lw=1, ls="--", alpha=0.4)

                    if j != 0:
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", left=False)
                    else:
                        ax.set_ylabel(label1, fontsize=10)
                    if i == n - 1:
                        ax.set_xlabel(label2, fontsize=10)
        plt.subplots_adjust(hspace=0.0, wspace=0)
        figname = f"{key}_w_comp.png"
        logging.info(f"Saving figure to {figname}")
        fig.savefig(figname, bbox_inches="tight", dpi=150, transparent=True)


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        wfit_files = args.get("WFIT_SUMMARY")
        if wfit_files:
            df_all = make_summary_file(wfit_files)
            for name, df in df_all.groupby("name"):
                if df.shape[0] > 1:
                    plot_single_file(name, df)
                else:
                    logging.info(f"Gruop {name} has df shape {str(df.shape)}")
            plot_all_files(df_all)
            plot_scatter_comp(df_all)

        logging.info(f"Finishing gracefully")
    except Exception as e:
        logging.exception(str(e))
        raise e
