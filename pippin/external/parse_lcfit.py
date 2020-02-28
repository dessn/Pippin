import os

import numpy as np
import pandas as pd
import sys
import argparse
import logging
import yaml
from astropy.cosmology import FlatLambdaCDM


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


def add_muref(df, filename, alpha=0.14, beta=3.1, om=0.311, h0=70, MB=-19.361):
    # TODO: FInd alpha beta om, h0 better
    cols = ["zHD", "x1", "mB", "c"]
    for c in cols:
        if c not in df.columns:
            logging.exception(f"Filename {filename} has no column {c}, has {df.columns}")
    cosmo = FlatLambdaCDM(h0, om)
    cosmo_dist_mod = cosmo.distmod(df["zHD"]).value
    obs_dist_mod = df["mB"] + alpha * df["x1"] - beta * df["c"] - MB
    diff = obs_dist_mod - cosmo_dist_mod
    df["__MUDIFF"] = diff


def load_file(infile, outfile):

    logging.info(f"Attempting to load in original file {infile}")
    df = pd.read_csv(infile, delim_whitespace=True, comment="#")

    df.replace(-999, np.nan, inplace=True)
    add_muref(df, infile)
    df.to_csv(outfile, index=False, float_format="%0.5f")
    logging.info(f"Saved dataframe from {infile} to {outfile}")
    return df


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

        data_dfs = [load_file(f, fo) for f, fo in zip(args.get("DATA_FITRES_INPUT", []), args.get("DATA_FITRES_PARSED", []))]
        sim_dfs = [load_file(f, fo) for f, fo in zip(args.get("SIM_FITRES_INPUT", []), args.get("SIM_FITRES_PARSED", []))]

        logging.info(f"Finishing gracefully")

    except Exception as e:
        logging.exception(str(e))
        raise e
