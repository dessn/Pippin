import shutil
import gzip
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


def blind_df(df, args):
    to_blind = args.get("BLIND", [])
    np.random.seed(0)
    for p in to_blind:
        if p in df.columns:
            logging.info(f"Blinding columns {p} in dataframe")
            df[p] += np.random.normal(scale=0.2, size=1000)[111]
    return df


def save_blind(df, args, output):
    df2 = blind_df(df, args)
    df2.to_csv(output, index=False, float_format="%0.5f")


def make_summary_file(wfit_files, args):
    logging.info("Creating summary file")
    all_output_csv = args.get("WFIT_SUMMARY_OUTPUT")

    df_all = None
    for f in wfit_files:
        logging.debug(f"Reading in wfit_summary {f}")
        df = pd.read_csv(f, delim_whitespace=True, comment="#").drop(columns=["VARNAMES:", "ROW"])
        name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        df["name"] = name
        logging.debug(f"Read {f}, contents are: {df}")
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])
    save_blind(df_all, args, all_output_csv)


def parse_fitres_files(args):
    fitres_input = args.get("FITRES_INPUT")
    fitres_output = args.get("FITRES_PARSED")
    logging.debug(f"FITRES_INPUT is {fitres_input}")
    logging.debug(f"FITRES_PARSED is {fitres_output}")

    for fin, fout in zip(fitres_input, fitres_output):
        logging.info(f"Parsing fitres file {fin} into {fout}")
        shutil.copy(fin, fout)


def parse_m0diffs(args):
    m0diffs = args.get("M0DIFF_INPUTS")
    m0diff_out = args.get("M0DIFF_PARSED")

    # logging.debug(f"M0DIFF_INPUTS is {m0diffs}")
    # logging.debug(f"M0DIFF_PARSED is {m0diff_out}")

    df_all = None
    for name, num, muopt, muopt_num, fitopt, fitopt_num, path in m0diffs:
        logging.debug(f"Parsing m0diff {path}")
        df = pd.read_csv(path, delim_whitespace=True, comment="#")

        ol_ref = np.NaN
        w_ref = np.NaN

        with gzip.open(path, "rt") as f:
            for line in f.read().splitlines():
                if "Omega_DE(ref)" in line:
                    ol_ref = float(line.strip().split()[-1])
                if "w_DE(ref)" in line:
                    w_ref = float(line.strip().split()[-1])

        df["name"] = name
        df["sim_num"] = num
        df["muopt"] = muopt
        df["muopt_num"] = muopt_num
        df["fitopt"] = fitopt
        df["fitopt_num"] = fitopt_num
        df["ol_ref"] = ol_ref
        df["w_ref"] = w_ref

        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    logging.info(f"Saving m0diffs to {m0diff_out}")
    save_blind(df_all, args, m0diff_out)


def parse_fitres(args):
    m0diffs = args.get("M0DIFF_INPUTS")
    fites_out = args.get("M0DIFF_PARSED").replace("m0diffs", "fitres")

    df_all = None

    for name, num, muopt, muopt_num, fitopt, fitopt_num, path in m0diffs:
        path = path.replace(".M0DIF", ".FITRES")
        logging.info(f"Attempting to parse file {path}")
        use_cols = ["CID", "IDSURVEY", "zHD", "mb", "x1", "c", "MU", "MUERR"]

        df = pd.read_csv(path, delim_whitespace=True, comment="#", usecols=use_cols)
        df["name"] = name
        df["sim_num"] = num
        df["muopt"] = muopt
        df["muopt_num"] = muopt_num
        df["fitopt"] = fitopt
        df["fitopt_num"] = fitopt_num

        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    logging.info(f"Saving combined fitres to {fites_out}")
    save_blind(df_all, args, fites_out)


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        wfit_files = args.get("WFIT_SUMMARY_INPUT")
        if wfit_files:
            make_summary_file(wfit_files, args)

        parse_fitres_files(args)
        parse_m0diffs(args)
        parse_fitres(args)
        logging.info(f"Finishing gracefully")
    except Exception as e:
        logging.exception(str(e))
        raise e
