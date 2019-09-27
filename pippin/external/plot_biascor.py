import numpy as np
from chainconsumer import ChainConsumer
import pandas as pd
import sys
import argparse
import os
import logging


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
    output_file = "all_biascor.png"

    c = ChainConsumer()
    labels = [r"$\Omega_m$", "$w$", r"$\sigma_{int}$"]
    for f, df in zip(source_files, inputs):
        name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        means = [df["OM"].mean(), df["w"].mean(), df["sigint"].mean()]
        if df.shape[0] < 2:
            name += " (showing mean error)"
            cov = np.diag([df["OM_sig"].mean() ** 2, df["wsig_marg"].mean() ** 2, 0.01 ** 2])
        else:
            name += " (showing scatter error)"
            cov = np.diag([df["OM"].std() ** 2, df["w"].std() ** 2, df["sigint"].std() ** 2])
        c.add_covariance(means, cov, parameters=labels, name=name.replace("_", "\\_"))
    c.plotter.plot_summary(errorbar=True, filename=output_file)


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
        with open(args.donefile, "w") as f:
            f.write("FAILURE")
