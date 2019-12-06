import os

import numpy as np
import pandas as pd
import sys
import argparse
import logging
import matplotlib.pyplot as plt
import yaml
from scipy.stats import binned_statistic, moment


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
    parser.add_argument("--donefile", help="Path of done file", type=str, default="plot_histogram.done")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config["donefile"] = args.donefile
    config.update(config["HISTOGRAM"])
    return config


def load_file(file):
    name = file.split("/")[-4]
    newfile = name + ".csv.gz"
    if os.path.exists(newfile):
        logging.info(f"Loading existing csv.gz file: {newfile}")
        return pd.read_csv(newfile), name
    else:
        logging.info(f"Attempting to load in original file {file}")
        df = pd.read_csv(file, delim_whitespace=True, comment="#")
        # df2 = df[["x1", "c", "zHD", "FITPROB", "SNRMAX1", "cERR", "x1ERR", "PKMJDERR", "TYPE"]]
        df.to_csv(newfile, index=False, float_format="%0.5f")
        logging.info(f"Saved dataframe from {file} to {newfile}")
        return df, name


def plot_histograms(data, sims, types):

    cols = ["x1", "c", "zHD", "FITPROB", "SNRMAX1", "SNRMAX2", "SNRMAX3", "cERR", "x1ERR", "PKMJDERR"]
    restricted = ["SNRMAX1", "SNRMAX2", "SNRMAX3"]

    for c in restricted:
        for x in data + sims:
            x[0].loc[x[0][c] < -10, c] = -9

    ncols = (len(cols) + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(1 + 2 * ncols, 5), gridspec_kw={"wspace": 0.3, "hspace": 0.3})
    for c, ax in zip(cols, axes.flatten()):
        u = 0.95 if c in restricted else 0.99
        minv = min([x[0][c].quantile(0.01) for x in data + sims])
        maxv = max([x[0][c].quantile(u) for x in data + sims])
        bins = np.linspace(minv, maxv, 20)  # Keep binning uniform.
        bc = 0.5 * (bins[1:] + bins[:-1])

        for i, (d, n) in enumerate(data):
            hist, _ = np.histogram(d[c], bins=bins)
            err = np.sqrt(hist)
            area = (bins[1] - bins[0]) * hist.sum()
            delta = (bc[1] - bc[0]) / 20
            ax.errorbar(bc + i * delta, hist / area, yerr=err / area, fmt="o", ms=2, elinewidth=0.75, label=n)

        for s, n in sims:
            mask = np.isin(s["TYPE"], types)
            ia = s[mask]
            nonia = s[~mask]

            hist, _ = np.histogram(s[c], bins=bins)
            area = (bins[1] - bins[0]) * hist.sum()

            ax.hist(s[c], bins=bins, histtype="step", weights=np.ones(s[c].shape) / area, label=n)
            if len(sims) < 3 and nonia.shape[0] > 0 and len(data) == 1:
                ax.hist(ia[c], bins=bins, histtype="step", weights=np.ones(ia[c].shape) / area, linestyle="--", label=n + " Ia only")
                ax.hist(nonia[c], bins=bins, histtype="step", weights=np.ones(nonia[c].shape) / area, linestyle=":", label=n + " CC only")

        ax.set_xlabel(c)
    plt.legend(bbox_to_anchor=(-3, 2.3, 4.0, 0.2), loc="lower left", mode="expand", ncol=2, frameon=False)
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig("hist.png", bbox_inches="tight", dpi=600)


def get_means_and_errors(x, y, bins):
    means, *_ = binned_statistic(x, y, bins=bins, statistic="mean")
    err, *_ = binned_statistic(x, y, bins=bins, statistic=lambda x: np.std(x) / np.sqrt(x.size))

    std, *_ = binned_statistic(x, y, bins=bins, statistic=lambda x: np.std(x))
    std_err, *_ = binned_statistic(
        x, y, bins=bins, statistic=lambda x: np.sqrt((1 / x.size) * (moment(x, 4) - (((x.size - 3) / (x.size - 1)) * np.var(x) ** 2))) / (2 * np.std(x))
    )
    return means, err, std, std_err


def plot_redshift_evolution(data, sims, types):

    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0.0, "wspace": 0.4})
    cols = ["x1", "c"]

    for c, row in zip(cols, axes.T):
        ax0, ax1 = row

        minv = min([x[0]["zHD"].min() for x in data + sims])
        maxv = max([x[0]["zHD"].max() for x in data + sims])
        bins = np.linspace(minv, maxv, 10)  # Keep binning uniform.
        bc = 0.5 * (bins[1:] + bins[:-1])
        lim = (bc[0] - 0.02 * (bc[-1] - bc[0]), bc[-1] + 0.02 * (bc[-1] - bc[0]))
        for d, n in data:
            means, err, std, std_err = get_means_and_errors(d["zHD"], d[c], bins=bins)
            ax0.errorbar(bc, means, yerr=err, fmt="o", ms=2, elinewidth=0.75, zorder=20, label=n)
            ax1.errorbar(bc, std, yerr=std_err, fmt="o", ms=2, elinewidth=0.75, zorder=20, label=n)

        for sim, n in sims:
            mask = np.isin(sim["TYPE"], types)
            ia = sim[mask]
            nonia = sim[~mask]

            has_nonia = nonia.shape[0] > 0
            if has_nonia:
                groups = [(sim, "-", 10, " all"), (ia, "--", 3, " Ia"), (nonia, ":", 2, " CC")]
            else:
                groups = [(sim, "-", 10, "")]

            for s, ls, z, n2 in groups:
                if s.shape[0] < 100:
                    continue
                means, err, std, std_err = get_means_and_errors(s["zHD"], s[c], bins=bins)
                ax0.plot(bc, means, ls=ls, zorder=z, label=n + n2)
                ax0.fill_between(bc, means - err, means + err, alpha=0.1, zorder=z)
                ax1.plot(bc, std, ls=ls, zorder=z, label=n + n2)
                ax1.fill_between(bc, std - std_err, std + std_err, alpha=0.1, zorder=z)

        ax0.set_ylabel(f"Mean {c}")
        ax1.set_ylabel(f"Std {c}")
        ax0.set_xlim(*lim)
        ax1.set_xlim(*lim)
    axes[1, 0].set_xlabel("z")
    axes[1, 1].set_xlabel("z")
    plt.legend(bbox_to_anchor=(-1.2, 2, 2.1, 0.2), loc="lower left", mode="expand", ncol=2, frameon=False)
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig("redshift.png", bbox_inches="tight", dpi=150, transparent=True)


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        if not args.get("DATA_FITRES"):
            logging.warning("Warning, no data files specified")
        if not args.get("SIM_FITRES"):
            logging.warning("Warning, no sim files specified")
        if not args.get("IA_TYPES"):
            logging.warning("Warning, no Ia types specified, assuming 1 and 101.")
            args["IA_TYPES"] = [1, 101]

        data_dfs = [load_file(f) for f in args.get("DATA_FITRES", [])]
        sim_dfs = [load_file(f) for f in args.get("SIM_FITRES", [])]

        for df, n in data_dfs + sim_dfs:
            df.replace(-999, np.nan, inplace=True)

        plot_histograms(data_dfs, sim_dfs, args["IA_TYPES"])
        plot_redshift_evolution(data_dfs, sim_dfs, args["IA_TYPES"])

        logging.info(f"Writing success to {args['donefile']}")
        with open(args["donefile"], "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        logging.exception(str(e))
        logging.error(f"Writing failure to file {args['donefile']}")
        with open(args["donefile"], "w") as f:
            f.write("FAILURE")
