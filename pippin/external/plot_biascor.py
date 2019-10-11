import numpy as np
from chainconsumer import ChainConsumer
import pandas as pd
import sys
import argparse
import os
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def setup_logging():
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler("plot_biascor.log")])
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="The base to use for paramnames Eg /path/SN_CMB_OMW_ALL", nargs="*", type=str)
    parser.add_argument("-b", "--blind", help="Blind these parameters", nargs="*", type=str, default=[])
    parser.add_argument("-d", "--donefile", help="Path of done file", type=str, default="done2.txt")
    args = parser.parse_args()

    logging.info(str(args))
    return args


def load_file(file, args):
    np.random.seed(123)
    logging.info(f"Loading in file {file}")
    df = pd.read_csv(file)
    if "w" in args.blind:
        df["w"] += np.random.normal(loc=0, scale=0.2, size=1000)[343]
    if "om" in args.blind:
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


def plot_all_files(source_files, inputs):
    logging.info("Plotting all files")
    output_file = "all_biascor.png"

    c = ChainConsumer()
    labels = [r"$\Omega_m$", "$w$", r"$\sigma_{int}$"]
    data = []
    df_all = None
    for f, df in zip(source_files, inputs):
        name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        means = [df["OM"].mean(), df["w"].mean(), df["sigint"].mean()]
        if df.shape[0] < 2:
            name2 = name + " (showing mean error)"
            cov = np.diag([df["OM_sig"].mean() ** 2, df["wsig_marg"].mean() ** 2, 0.01 ** 2])

        else:
            name2 = name + " (showing scatter error)"
            cov = np.diag([df["OM"].std() ** 2, df["w"].std() ** 2, df["sigint"].std() ** 2])

        df["name"] = name
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

        data.append([name, df["w"].mean(), df["w"].std(), df["wsig_marg"].mean()])
        c.add_covariance(means, cov, parameters=labels, name=name2.replace("_", "\\_"))
    wdf = pd.DataFrame(data, columns=["name", "mean_w", "scatter_mean_w", "mean_std_w"])
    wdf.to_csv(output_file.replace(".png", ".csv"), index=False, float_format="%0.4f")
    df_all.to_csv(output_file.replace(".png", "_individual.csv"), index=False, float_format="%0.4f")
    c.plotter.plot_summary(errorbar=True, filename=output_file)

    plot_scatter_comp(df_all)


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
        lim = (min_w, max_w)

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
                    ax.scatter(a1, a2, s=2, c=c, cmap="viridis_r", vmin=-0.05, vmax=0.2)
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
        if args.basename:
            inputs = [load_file(w, args) for w in args.basename]

            for file, df in zip(args.basename, inputs):
                if df.shape[0] > 1:
                    plot_single_file(file, df)
                else:
                    logging.info(f"File {file} has df shape {str(df.shape)}")
            plot_all_files(args.basename, inputs)

        with open(args.donefile, "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        logging.exception(str(e))
        logging.error("Writing failure to file")
        with open(args.donefile, "w") as f:
            f.write("FAILURE")
