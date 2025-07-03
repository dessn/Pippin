import numpy as np
import pandas as pd
import sys
import argparse
import logging
import matplotlib.pyplot as plt
import yaml
from scipy.stats import binned_statistic, moment
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def setup_logging():
    fmt = "[%(levelname)8s |%(funcName)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[handler, logging.FileHandler("plot_biascor.log")],
    )
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
    if config.get("FIELDS") is None:
        config["FIELDS"] = [
            ["X3", "C3"],
            ["E1", "E2", "C1", "C2", "S1", "S2", "X1", "X2"],
        ]
    return config


def load_file(file):
    logging.info(f"Loading existing csv.gz file: {file}")
    return pd.read_csv(file)


def plot_efficiency(data_all, sims, fields):
    for i, sim in enumerate(sims):
        fig, axes = plt.subplots(
            len(fields), 3, figsize=(12, 1 + 2 * len(fields)), squeeze=False
        )
        cols = ["HOST_MAG_i", "HOST_MAG_r", "zHD"]

        field_eff = {}
        for field, row in zip(fields, axes):
            data = data_all[np.isin(data_all["FIELD"], field)]
            s = sim[np.isin(sim["FIELD"], field)]

            for c, ax in zip(cols, row):
                if c == "zHD":
                    bins = np.arange(0.15, 1.55, 0.1)
                else:
                    bins = np.arange(15.5, 30.5, 1)

                title = " ".join(field)
                if len(title) > 20:
                    title = title[:20] + "..."
                ax.set_title(title)

                minv = min([x[c].quantile(0.01) for x in [data, s]])
                maxv = max([x[c].quantile(0.99) for x in [data, s]])
                bins2 = np.linspace(minv, maxv, 200)  # Keep binning uniform.
                bc = 0.5 * (bins[1:] + bins[:-1])
                bc2 = 0.5 * (bins2[1:] + bins2[:-1])

                hist_data, _ = np.histogram(data[c], bins=bins)
                hist_data[hist_data < 5] = 0
                hist_data2, _ = np.histogram(data[c], bins=bins2)
                err_data = np.sqrt(hist_data)

                hist_sim, _ = np.histogram(s[c], bins=bins)
                mask_sim_zero = hist_sim == 0
                hist_sim2, _ = np.histogram(s[c], bins=bins2)
                mask_sim2_zero = hist_sim2 == 0

                hist_sim[mask_sim_zero] = 1
                hist_sim2[mask_sim2_zero] = 1

                err_sim = np.sqrt(hist_sim)

                ratio = hist_data / hist_sim
                ratio = ratio / ratio.max()
                ratio2 = hist_data2 / hist_sim2
                ratio2 = ratio2 / ratio2.max()

                ratio3 = np.concatenate((ratio2, np.zeros(100)))
                bc3 = interp1d(
                    np.arange(bc2.size),
                    bc2,
                    bounds_error=False,
                    fill_value="extrapolate",
                )(np.arange(bc2.size + 20))
                smoothed_ratio = gaussian_filter(ratio3, sigma=4, mode="nearest")[:-80]
                smoothed_ratio = smoothed_ratio / smoothed_ratio.max()

                err = (
                    np.sqrt((err_data / hist_data) ** 2 + (err_sim / hist_sim) ** 2)
                    * ratio
                )

                ddf = pd.DataFrame(
                    {
                        c: bc,
                        "Ndata": hist_data,
                        "Nsim": hist_sim,
                        "eff": ratio,
                        "eff_err": err,
                    }
                )
                ddf.to_csv(f"eff_{c}.csv", index=False, float_format="%0.4f")

                ax.plot(bc, ratio, linewidth=0.5)
                ax.plot(bc2, ratio2, linewidth=0.5, color="k", alpha=0.5, ls=":")
                ax.fill_between(bc, ratio - err, ratio + err, alpha=0.3)

                ax.plot(bc3, smoothed_ratio, c="k", lw=1)

                ax.plot(bc, hist_data / hist_data.max(), c="r", ls="--", lw=0.5)
                ax.plot(bc, hist_sim / hist_sim.max(), c="g", ls="--", lw=0.5)
                ax.set_ylim(0, 1.1)
                ax.set_xlabel(c)

                y = ratio.copy()
                y[y < 0.02] = 0

                if field_eff.get(c) is None:
                    field_eff[c] = []

                field_eff[c].append((bc, y))

        save_efficiency_file(field_eff, fields)

        fig.tight_layout()
        fig.savefig(
            f"efficiency_{i}.png", bbox_inches="tight", dpi=150, transparent=True
        )


def save_efficiency_file(field_eff, fields):
    labels = field_eff.keys()
    name_map = {
        "HOST_MAG_i": "i_obs",
        "HOST_MAG_r": "r_obs",
        "HOST_MAG_z": "z_obs",
        "HOST_MAG_g": "g_obs",
        "zHD": "ZTRUE",
    }
    for c in labels:
        with open(f"efficiency_{c}.dat", "w") as f:
            header = "OPT_EXTRAP: 1\n\n"
            f.write(header)

            for field, (xs, ys) in zip(fields, field_eff[c]):
                header2 = f"""FIELDLIST: {"+".join(field)}
VARNAMES: {name_map[c]} HOSTEFF
"""
                f.write(header2)
                for x, y in zip(xs, ys):
                    f.write(f"HOSTEFF:  {x:7.2f} {y:8.3f}\n")
                f.write("ENDMAP:\n\n")


def plot_efficiency2d(data_all, sims, fields):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for i, sim in enumerate(sims):
        fig, axes = plt.subplots(3, len(fields), figsize=(15, 8), squeeze=False)
        axes = np.atleast_2d(axes)

        ci = "HOST_MAG_i"
        cr = "HOST_MAG_i-r"

        for field, ax in zip(fields, axes.T):
            ff = " ".join(field)
            data = data_all[np.isin(data_all["FIELD"], field)]
            s = sim[np.isin(sim["FIELD"], field)]

            threshold = 1

            min_i = min([x[ci].quantile(0.01) for x in [data, s]])
            max_i = max([x[ci].quantile(0.99) for x in [data, s]])
            min_r = min([x[cr].quantile(0.01) for x in [data, s]])
            max_r = max([x[cr].quantile(0.99) for x in [data, s]])

            bins_i = np.linspace(min_i, max_i, 16)
            bins_r = np.linspace(min_r, max_r, 4)

            hist_data, _, _ = np.histogram2d(data[ci], data[cr], (bins_i, bins_r))
            hist_data[hist_data < threshold] = np.nan

            hist_sim, _, _ = np.histogram2d(s[ci], s[cr], bins=(bins_i, bins_r))
            hist_sim[hist_sim < threshold] = np.nan

            ratio = hist_data / hist_sim
            # ratio = ratio / ratio[np.isfinite(ratio)].max()

            im = ax[0].imshow(
                hist_data.T, origin="lower", extent=[min_i, max_i, min_r, max_r]
            )
            ax[0].set_title("Data " + ff)
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            im = ax[1].imshow(
                hist_sim.T, origin="lower", extent=[min_i, max_i, min_r, max_r]
            )
            ax[1].set_title("Sim " + ff)
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            im = ax[2].imshow(
                ratio.T, origin="lower", extent=[min_i, max_i, min_r, max_r]
            )
            ax[2].set_title("Ratio " + ff)
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            ax[2].set_xlabel(ci)
            for a in ax:
                a.set_ylabel(cr)

        fig.tight_layout()
        fig.savefig(
            f"efficiency2d_{i}.png", bbox_inches="tight", dpi=150, transparent=True
        )


def get_means_and_errors(x, y, bins):
    means, *_ = binned_statistic(x, y, bins=bins, statistic="mean")
    err, *_ = binned_statistic(
        x, y, bins=bins, statistic=lambda x: np.std(x) / np.sqrt(x.size)
    )

    std, *_ = binned_statistic(x, y, bins=bins, statistic=lambda x: np.std(x))
    std_err, *_ = binned_statistic(
        x,
        y,
        bins=bins,
        statistic=lambda x: np.sqrt(
            (1 / x.size)
            * (moment(x, 4) - (((x.size - 3) / (x.size - 1)) * np.var(x) ** 2))
        )
        / (2 * np.std(x)),
    )
    return means, err, std, std_err


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        if not args.get("DATA_FITRES_PARSED"):
            logging.warning("Warning, no data files specified")
        if not args.get("SIM_FITRES_PARSED"):
            logging.warning("Warning, no sim files specified")

        data_dfs = [load_file(f) for f in args.get("DATA_FITRES_PARSED", [])]
        sim_dfs = [load_file(f) for f in args.get("SIM_FITRES_PARSED", [])]

        if "HOST_MAG_i" not in sim_dfs[0].columns:
            logging.info("HOST_MAG_i not in output fitres, not computing efficiencies")
        else:
            if len(data_dfs) > 1:
                logging.info(
                    "Please specify only one data file if you want to calculate efficiency"
                )
            else:
                for d in data_dfs + sim_dfs:
                    d["HOST_MAG_i-r"] = d["HOST_MAG_i"] - d["HOST_MAG_r"]
                plot_efficiency(data_dfs[0], sim_dfs, args["FIELDS"])

        logging.info("Finishing gracefully")

    except Exception as e:
        logging.exception(e, exc_info=True)
        raise e
