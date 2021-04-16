import os

import numpy as np
import pandas as pd
import sys
import argparse
import logging
import matplotlib.pyplot as plt
import yaml
from astropy.cosmology import FlatLambdaCDM
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
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config.update(config["LCFIT"])
    return config


def load_file(file):
    name = file.split(".")[0]
    logging.info(f"Loading existing csv.gz file: {file}")
    return pd.read_csv(file), name


def plot_histograms(data, sims, types, figname):
    cols = [
        "x1",
        "c",
        "zHD",
        "FITPROB",
        "FITCHI2",
        "cERR",
        "x1ERR",
        "PKMJDERR",
        "HOST_LOGMASS",
        "SNRMAX1",
        "SNRMAX2",
        "SNRMAX3",
        "SNRMAX_g",
        "SNRMAX_r",
        "SNRMAX_i",
        "SNRMAX_z",
        "NDOF",
        "chi2_g",
        "chi2_r",
        "chi2_i",
        "chi2_z",
        "__MUDIFF",
    ]
    restricted = ["FITCHI2", "SNRMAX1", "SNRMAX2", "SNRMAX3", "SNRMAX_g", "SNRMAX_r", "SNRMAX_i", "SNRMAX_z", "chi2_g", "chi2_r", "chi2_i", "chi2_z"]
    logs = ["SNRMAX1", "SNRMAX2", "SNRMAX3", "SNRMAX_g", "SNRMAX_r", "SNRMAX_i", "SNRMAX_z", "FITCHI2", "chi2_g", "chi2_r", "chi2_i", "chi2_z", "__MUDIFF"]

    cs = ["#1976D2", "#FB8C00", "#8BC34A", "#E53935", "#4FC3F7", "#43A047", "#F2D026", "#673AB7", "#FFB300", "#E91E63", "#F2D026"] * 3
    cols = [c for c in cols if c in data[0][0].columns]

    for c in restricted:
        for x in data + sims:
            if c in cols:
                x[0].loc[x[0][c] < -10, c] = -9

    ncols = (len(cols) + 3) // 3
    fig, axes = plt.subplots(3, ncols, figsize=(1 + 2.5 * ncols, 8), gridspec_kw={"wspace": 0.13, "hspace": 0.4})
    for ax in axes.flatten():
        ax.set_axis_off()

    for c, ax in zip(cols, axes.flatten()):
        ax.set_axis_on()
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

        lw = 1 if len(sims) < 3 else 0.5
        for index, (s, n) in enumerate(sims):
            mask = np.isin(s["TYPE"], types)
            ia = s[mask]
            nonia = s[~mask]

            hist, _ = np.histogram(s[c], bins=bins)
            area = (bins[1] - bins[0]) * hist.sum()

            ax.hist(s[c], bins=bins, histtype="step", weights=np.ones(s[c].shape) / area, label=n, linewidth=lw, color=cs[index])
            if len(sims) == 1 and nonia.shape[0] > 10 and len(data) == 1:
                logging.info(f"Nonia shape is {nonia.shape}")
                ax.hist(ia[c], bins=bins, histtype="step", weights=np.ones(ia[c].shape) / area, linestyle="--", label=n + " Ia only", linewidth=1)
                ax.hist(nonia[c], bins=bins, histtype="step", weights=np.ones(nonia[c].shape) / area, linestyle=":", label=n + " CC only", linewidth=1)

        if "MUDIFF" in c:
            ax.set_xlabel("FAKE MUDIFF")
        else:
            ax.set_xlabel(c)
        if c in logs:
            ax.set_yscale("log")
            ax.tick_params(axis="y", which="both", labelsize=2)
            labels = ["" for item in ax.get_yticklabels()]
            ax.set_yticklabels(labels)

        # Add the reduced chi2 value if there are only one data and one sim
        if len(sims) < 3 and len(data) == 1:
            data_col = data[0][0][c]
            data_hist, _ = np.histogram(data_col, bins=bins)
            data_err = 1 / np.sqrt(data_hist)
            data_dist, _ = np.histogram(data_col, bins=bins, density=True)

            for i, (s, n) in enumerate(sims):
                sim_col = s[c]

                sim_hist, _ = np.histogram(sim_col, bins=bins)
                sim_err = 1 / np.sqrt(data_hist)
                sim_dist, _ = np.histogram(sim_col, bins=bins, density=True)

                dist_error = np.sqrt((data_dist * data_err) ** 2 + (sim_dist * sim_err) ** 2)
                dist_diff = data_dist - sim_dist

                chi2 = np.nansum(((dist_diff / dist_error) ** 2))
                ndof = len(bc)
                red_chi2 = chi2 / ndof

                ax.text(
                    0.99,
                    0.99 - 0.1 * i,
                    f"{chi2:0.1f}/{ndof:d}={red_chi2:0.1f}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                    color=cs[i],
                    fontsize=8,
                )

    handles, labels = ax.get_legend_handles_labels()
    bb = (fig.subplotpars.left, fig.subplotpars.top + 0.02, fig.subplotpars.right - fig.subplotpars.left, 0.1)

    for ax in axes.flatten():
        ax.set_yticklabels([])

    fig.legend(handles, labels, loc="upper center", ncol=4, mode="expand", frameon=False, bbox_to_anchor=bb, borderaxespad=0.0, bbox_transform=fig.transFigure)
    # plt.legend(bbox_to_anchor=(-3, 2.3, 4.0, 0.2), loc="lower left", mode="expand", ncol=3, frameon=False)
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig(figname, bbox_inches="tight", dpi=600)


def get_means_and_errors(x, y, bins):
    x = np.array(x)
    y = np.array(y)
    means, *_ = binned_statistic(x, y, bins=bins, statistic="mean")
    err, *_ = binned_statistic(x, y, bins=bins, statistic=lambda x: np.std(x) / np.sqrt(len(x)))

    std, *_ = binned_statistic(x, y, bins=bins, statistic=lambda x: np.std(x))
    std_err, *_ = binned_statistic(
        x,
        y,
        bins=bins,
        statistic=lambda x: np.nan
        if len(x) < 20
        else np.sqrt((1 / len(x)) * (moment(x, 4) - (((len(x) - 3) / (len(x) - 1)) * np.var(x) ** 2))) / (2 * np.std(x)),
    )
    return means, err, std, std_err


def plot_redshift_evolution(data, sims, types, figname):
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0.0, "wspace": 0.4})
    cols = ["x1", "c"]

    for c, row in zip(cols, axes.T):
        ax0, ax1 = row

        minv = min([x[0]["zHD"].min() for x in data + sims])
        maxv = max([x[0]["zHD"].max() for x in data + sims])
        if not np.isfinite(minv) or not np.isfinite(maxv):
            continue

        bins = np.linspace(minv, maxv, 10)  # Keep binning uniform.
        bc = 0.5 * (bins[1:] + bins[:-1])
        lim = (bc[0] - 0.02 * (bc[-1] - bc[0]), bc[-1] + 0.02 * (bc[-1] - bc[0]))
        for d, n in data:
            if d.shape[0] == 0:
                continue
            means, err, std, std_err = get_means_and_errors(d["zHD"], d[c], bins=bins)
            ax0.errorbar(bc, means, yerr=err, fmt="o", ms=2, elinewidth=0.75, zorder=20, label=n)
            ax1.errorbar(bc, std, yerr=std_err, fmt="o", ms=2, elinewidth=0.75, zorder=20, label=n)

        for sim, n in sims:
            if sim.shape[0] == 0:
                continue
            mask = np.isin(sim["TYPE"], types)
            ia = sim[mask]
            nonia = sim[~mask]

            has_nonia = nonia.shape[0] > 0
            if has_nonia and len(sims) == 1:
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
    fig.savefig(figname, bbox_inches="tight", dpi=150, transparent=True)


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        if not args.get("DATA_FITRES_PARSED"):
            logging.warning("Warning, no data files specified")
        if not args.get("SIM_FITRES_PARSED"):
            logging.warning("Warning, no sim files specified")
        if not args.get("IA_TYPES"):
            logging.warning("Warning, no Ia types specified, assuming 1 and 101.")
            args["IA_TYPES"] = [1, 101]

        data_dfs = [load_file(f) for f in args.get("DATA_FITRES_PARSED", [])]
        sim_dfs = [load_file(f) for f in args.get("SIM_FITRES_PARSED", [])]

        data_masks = [d["FITPROB"] > 0.05 for d, _ in data_dfs]
        sim_masks = [d["FITPROB"] > 0.05 for d, _ in sim_dfs]
        data_dfs = [(d[0].loc[m, :], d[1]) for d, m in zip(data_dfs, data_masks)]
        sim_dfs = [(d[0].loc[m, :], d[1]) for d, m in zip(sim_dfs, sim_masks)]

        plot_histograms(data_dfs, sim_dfs, args["IA_TYPES"], "hist.png")
        plot_redshift_evolution(data_dfs, sim_dfs, args["IA_TYPES"], "redshift.png")

        fields = [(["X3", "C3"], "DEEP"), (["C1", "C2", "S1", "S2", "E1", "E2", "X1", "X2"], "SHALLOW")]
        for f, n in fields:
            data_masks = [np.isin(d["FIELD"], f) for d, _ in data_dfs]
            sim_masks = [np.isin(s["FIELD"], f) for s, _ in sim_dfs]

            masked_data_dfs = [(d[0].loc[m, :], d[1]) for d, m in zip(data_dfs, data_masks)]
            masked_sim_dfs = [(d[0].loc[m, :], d[1]) for d, m in zip(sim_dfs, sim_masks)]
            plot_histograms(masked_data_dfs, masked_sim_dfs, args["IA_TYPES"], f"hist_{n}.png")
            plot_redshift_evolution(masked_data_dfs, masked_sim_dfs, args["IA_TYPES"], f"redshift_{n}.png")

        logging.info(f"Finishing gracefully")

    except Exception as e:
        logging.exception(str(e))
        raise e
