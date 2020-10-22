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
import gzip


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


def load_file(file):
    df = pd.read_csv(file)
    return df


def plot_single_file(source_file, df):
    logging.info(f"Plotting single file {source_file}")
    name = os.path.basename(os.path.dirname(os.path.abspath(source_file)))
    output_file = name + "_wfit.png"
    logging.info(f"Creating wfit plot output to {output_file}")

    c = ChainConsumer()
    labels = [r"$\Omega_m$", "$w$", r"$\sigma_{int}$"]
    for index, row in df.iterrows():
        means = [row["omm"], row["w"], row["sigint"]]
        cov = np.diag([row["omm_sig"] ** 2, row["w_sig"] ** 2, 0.01 ** 2])
        c.add_covariance(means, cov, parameters=labels, name=f"Realisation {index}")
    c.plotter.plot_summary(errorbar=True, filename=output_file)
    del c


def plot_all_files(df_all):
    output_file = "all_biascor_results.png"
    logging.info(f"Plotting all fits to {output_file}")

    c = ChainConsumer()
    labels = [r"$\Omega_m$", "$w$", r"$\sigma_{int}$"]
    data = []
    for name, df in df_all.groupby("name"):
        means = [df["omm"].mean(), df["w"].mean(), df["sigint"].mean()]
        if df.shape[0] < 2:
            name2 = name + " (showing mean error)"
            cov = np.diag([df["omm_sig"].mean() ** 2, df["w_sig"].mean() ** 2, 0.01 ** 2])
        else:
            name2 = name + " (showing scatter error)"
            cov = np.diag([df["omm"].std() ** 2, df["w"].std() ** 2, df["sigint"].std() ** 2])
        c.add_covariance(means, cov, parameters=labels, name=name2.replace("_", "\\_"))
        data.append([name, df["w"].mean(), df["w"].std(), df["w_sig"].mean()])
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

        if n == 1:
            logging.info("Only one version found, nothing to scatter against")
            return
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


def make_hubble_plot(fitres_file, m0diff_file, prob_col_name, args):
    logging.info(f"Making Hubble plot from FITRES file {fitres_file} and M0DIF file {m0diff_file}")
    # Note that the fitres file has mu and fit 0, m0diff will have to select down to it

    name, sim_num, *_ = fitres_file.split("_")
    sim_num = int(sim_num)

    df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")
    dfm = pd.read_csv(m0diff_file)
    dfm = dfm[(dfm.name == name) & (dfm.sim_num == sim_num) & (dfm.muopt_num == 0) & (dfm.fitopt_num == 0)]

    from astropy.cosmology import FlatwCDM
    import numpy as np
    import matplotlib.pyplot as plt

    df.sort_values(by="zHD", inplace=True)
    dfm.sort_values(by="z", inplace=True)
    dfm = dfm[dfm["MUDIFERR"] < 10]

    ol = dfm.ol_ref.unique()[0]
    w = dfm.w_ref.unique()[0]
    if np.isnan(ol):
        logging.info("Setting ol = 0.689")
        ol = 0.689
    if np.isnan(w):
        logging.info("Setting w = -1")
        w = -1
    alpha = 0
    beta = 0
    sigint = 0
    gamma = r"$\gamma = 0$"
    scalepcc = "NA"
    num_sn_fit = df.shape[0]
    contam_data, contam_true = "", ""

    with gzip.open(fitres_file, "rt") as f:
        for line in f.read().splitlines():
            if "NSNFIT" in line:
                v = int(line.split("=", 1)[1].strip())
                num_sn_fit = v
                num_sn = f"$N_{{SN}} = {v}$"
            if "alpha0" in line and "=" in line and "+-" in line:
                alpha = r"$\alpha = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
            if "beta0" in line and "=" in line and "+-" in line:
                beta = r"$\beta = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
            if "sigint" in line and "iteration" in line:
                sigint = r"$\sigma_{\rm int} = " + line.split()[3] + "$"
            if "gamma" in line and "=" in line and "+-" in line:
                gamma = r"$\gamma = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
            if "CONTAM_TRUE" in line:
                v = max(0.0, float(line.split("=", 1)[1].split("#")[0].strip()))
                n = v * num_sn_fit
                contam_true = f"$R_{{CC, true}} = {v:0.4f} (\\approx {int(n)} SN)$"
            if "CONTAM_DATA" in line:
                v = max(0.0, float(line.split("=", 1)[1].split("#")[0].strip()))
                n = v * num_sn_fit
                contam_data = f"$R_{{CC, data}} = {v:0.4f} (\\approx {int(n)} SN)$"
            if "scalePCC" in line and "+-" in line:
                scalepcc = "scalePCC = $" + line.split("=")[-1].strip().replace("+-", r"\pm") + "$"
    prob_label = prob_col_name.replace("PROB_", "").replace("_", " ")
    label = "\n".join([num_sn, alpha, beta, sigint, gamma, scalepcc, contam_true, contam_data, f"Classifier = {prob_label}"])
    label = label.replace("\n\n", "\n").replace("\n\n", "\n")
    dfz = df["zHD"]
    zs = np.linspace(dfz.min(), dfz.max(), 500)
    distmod = FlatwCDM(70, 1 - ol, w).distmod(zs).value

    n_trans = 1000
    n_thresh = 0.05
    n_space = 0.3
    subsec = True
    if zs.min() > n_thresh:
        n_space = 0.01
        subsec = False
    z_a = np.logspace(np.log10(min(0.01, zs.min() * 0.9)), np.log10(n_thresh), int(n_space * n_trans))
    z_b = np.linspace(n_thresh, zs.max() * 1.01, 1 + int((1 - n_space) * n_trans))[1:]
    z_trans = np.concatenate((z_a, z_b))
    z_scale = np.arange(n_trans)

    def tranz(zs):
        return interp1d(z_trans, z_scale)(zs)

    if subsec:
        x_ticks = np.array([0.01, 0.02, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
        x_ticks_m = np.array([0.03, 0.04, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9])
    else:
        x_ticks = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
        x_ticks_m = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.9])
    mask = (x_ticks > z_trans.min()) & (x_ticks < z_trans.max())
    mask_m = (x_ticks_m > z_trans.min()) & (x_ticks_m < z_trans.max())
    x_ticks = x_ticks[mask]
    x_ticks_m = x_ticks_m[mask_m]
    x_tick_t = tranz(x_ticks)
    x_ticks_mt = tranz(x_ticks_m)

    fig, axes = plt.subplots(figsize=(7, 5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0})
    logging.info(f"Hubble plot prob colour given by column {prob_col_name}")

    if prob_col_name.upper().startswith("PROB"):
        df[prob_col_name] = df[prob_col_name].clip(0, 1)

    for resid, ax in enumerate(axes):
        ax.tick_params(which="major", direction="inout", length=4)
        ax.tick_params(which="minor", direction="inout", length=3)
        if resid:
            sub = df["MUMODEL"]
            sub2 = 0
            sub3 = distmod
            ax.set_ylabel(r"$\Delta \mu$")
            ax.tick_params(top=True, which="both")
            alpha = 0.2
            ax.set_ylim(-0.5, 0.5)
        else:
            sub = 0
            sub2 = -dfm["MUREF"]
            sub3 = 0
            ax.set_ylabel(r"$\mu$")
            ax.annotate(label, (0.98, 0.02), xycoords="axes fraction", horizontalalignment="right", verticalalignment="bottom", fontsize=8)
            alpha = 0.7

        ax.set_xlabel("$z$")
        if subsec:
            ax.axvline(tranz(n_thresh), c="#888888", alpha=0.4, zorder=0, lw=0.7, ls="--")

        if df[prob_col_name].min() >= 1.0:
            cc = df["IDSURVEY"]
            vmax = None
            color_prob = False
            cmap = "rainbow"
        else:
            cc = df[prob_col_name]
            vmax = 1.05
            color_prob = True
            cmap = "inferno"

        # Plot each point
        ax.errorbar(tranz(dfz), df["MU"] - sub, yerr=df["MUERR"], fmt="none", elinewidth=0.5, c="#AAAAAA", alpha=0.5 * alpha)
        h = ax.scatter(tranz(dfz), df["MU"] - sub, c=cc, s=1, zorder=2, alpha=alpha, vmax=vmax, cmap=cmap)

        if not args.get("BLIND", []):
            # Plot ref cosmology
            ax.plot(tranz(zs), distmod - sub3, c="k", zorder=-1, lw=0.5, alpha=0.7)

            # Plot m0diff
            ax.errorbar(tranz(dfm["z"]), dfm["MUDIF"] - sub2, yerr=dfm["MUDIFERR"], fmt="o", mew=0.5, capsize=3, elinewidth=0.5, c="k", ms=4)
        ax.set_xticks(x_tick_t)
        ax.set_xticks(x_ticks_mt, minor=True)
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(z_scale.min(), z_scale.max())

        if args.get("BLIND", []):
            ax.set_yticklabels([])
            ax.set_yticks([])
    if color_prob:
        cbar = fig.colorbar(h, ax=axes, orientation="vertical", fraction=0.1, pad=0.01, aspect=40)
        cbar.set_label("Prob Ia")

    fp = fitres_file.replace(".fitres.gz", ".png")
    logging.debug(f"Saving Hubble plot to {fp}")
    fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


def make_m0diff_plot(m0diff_file):
    logging.info("Making m0diff plot")
    dfm = pd.read_csv(m0diff_file)

    # Figure out how to break up the df, if needed
    group = []
    if dfm.shape[0] > 30:
        if len(dfm.name.unique()) > 5:
            group.append("name")
        if len(dfm.muopt_num.unique()) > 5:
            group.append("muopt")

    if group:
        dfg = dfm.groupby(group)
        n = len(dfg)
    else:
        dfg = [("", dfm)]
        n = 1

    if n > 6:
        ncols = 3
    else:
        ncols = 1
    nrows = (n + (ncols - 1)) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 1 + 1.5 * nrows), squeeze=False, sharex=True)
    axes = axes.flatten()

    for (name, df), ax in zip(dfg, axes):
        if name == "":
            label_cols = ["name", "muopt", "fitopt"]
        elif isinstance(name, str):
            label_cols = ["muopt", "fitopt"]
        else:
            label_cols = ["fitopt"]

        # Now group again and plot out the m0diff
        dfg2 = df.groupby(label_cols)
        for label, df2 in dfg2:
            if not isinstance(label, str):
                label = " ".join(list(label)).replace("_", " ")
            if "DEFAULT DEFAULT" in label.upper():
                ls = ":"
            else:
                ls = "-"
            ax.plot(df2.z, df2.MUDIF, label=label, ls=ls)

        if len(dfg2) > 10:
            ax.legend(bbox_to_anchor=(0.5, -0.1), ncol=2)
        else:
            ax.legend()
        ax.set_title(name)
        ax.set_xlabel("z")
        ax.set_ylabel("Delta mu")
    fig.savefig("biascor_m0diffs.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:

        # Plot wfit distributions
        wfit_file = args.get("WFIT_SUMMARY_OUTPUT")
        df_all = load_file(wfit_file)
        for name, df in df_all.groupby("name"):
            if df.shape[0] > 1:
                plot_single_file(name, df)
            else:
                logging.info(f"Group {name} has df shape {str(df.shape)}")
            plot_all_files(df_all)
            plot_scatter_comp(df_all)

        # Plot hubble diagrams
        m0diff_file = args.get("M0DIFF_PARSED")
        fitres_files = args.get("FITRES_PARSED")
        prob_cols = args.get("FITRES_PROB_COLS")
        for f, p in zip(fitres_files, prob_cols):
            make_hubble_plot(f, m0diff_file, p, args)

        # Plot M0diffs
        make_m0diff_plot(m0diff_file)

        # Plot tails

        logging.info(f"Finishing gracefully")
    except Exception as e:
        logging.exception(str(e), exc_info=True)
        raise e
