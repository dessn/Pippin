import numpy as np
import yaml
from chainconsumer import ChainConsumer
import pandas as pd
import sys
import argparse
import os
import logging


def load_params(file):
    assert os.path.exists(file), f"Paramnames file {file} does not exist"
    names, labels = [], []
    with open(file) as f:
        for line in f.read().splitlines():
            n, l = line.split(maxsplit=1)
            names.append(n.replace("*", ""))
            labels.append(l)
    return names, labels


def load_chains(files, all_cols, use_cols=None):
    header = ["weights", "likelihood"] + all_cols
    data = [pd.read_csv(f, delim_whitespace=True, header=None, names=header) for f in files]

    # Remove burn in by cutting off first 30%
    data = [d.iloc[int(d.shape[0] * 0.3) :, :] for d in data]

    combined = pd.concat(data)
    if use_cols is None:
        use_cols = all_cols
    weights = combined["weights"].values
    likelihood = combined["likelihood"].values
    chain = combined[use_cols].values
    return weights, likelihood, chain


def fail(msg, condition=True):
    if condition:
        logging.error(msg)
        raise ValueError(msg)


def get_chain_files(basename):
    folder = os.path.dirname(basename)
    logging.info(f"Looking for chains in folder {folder}")
    base = os.path.basename(basename)
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if base in f and f.endswith(".txt")]
    fail(f"No chain files found for {os.path.join(folder, basename)}", condition=len(files) == 0)
    logging.info(f"{len(files)} chains found for basename {basename}")
    return files


def setup_logging():
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler("parse_cosmomc.log")])
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def blind(chain, names, columns_to_blind, index=0):
    np.random.seed(123)
    for i, c in enumerate(columns_to_blind):
        logging.info(f"Blinding column {c}")
        try:
            ii = names.index(c)
            scale = np.random.normal(loc=1, scale=0.0, size=1000)[321 + i]
            offset = np.random.normal(loc=0, scale=0.2, size=1000)[343 + i + (index + 10)]
            chain[:, ii] = chain[:, ii] * scale + np.std(chain[:, ii]) * offset
        except ValueError as e:
            logging.warning(f"Cannot find blinding column {c} in list of names {names}")


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input yml file", type=str)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config.update(config["COSMOMC"])

    if config.get("NAMES") is not None:
        assert len(config["NAMES"]) == len(config["INPUT_FILES"]), (
            "You should specify one name per base file you pass in." + f" Have {len(config['FILES'])} base names and {len(config['NAMES'])} names"
        )
    return config


def parse_chains(basename, outname, args, index):

    logging.info("Loading in data from original CosmoMC files")
    param_file = os.path.join(basename) + ".paramnames"
    chain_files = get_chain_files(basename)
    names, labels = load_params(param_file)
    blind_params = args.get("BLIND")
    params = args.get("PARAMS")
    weights, likelihood, chain = load_chains(chain_files, names, params)
    if blind_params:
        blind(chain, params or names, blind_params, index=index)
    labels = [
        f"${l}" + (r"\ \mathrm{Blinded}" if blind_params is not None and u in blind_params else "") + "$"
        for u in params
        for l, n in zip(labels, names)
        if n == u
    ]

    # Turn into new df
    output_df = pd.DataFrame(np.vstack((weights, likelihood, chain.T)).T, columns=["_weight", "_likelihood"] + labels)
    output_df.to_csv(outname, float_format="%0.5f", index=False)

    logging.info(f"Chain for {basename} has shape {chain.shape}")
    logging.info(f"Labels for {basename} are {labels}")
    return weights, likelihood, labels, chain


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        if args.get("INPUT_FILES"):
            logging.info("Creating chain consumer object")

            biases = {}
            b = 1
            truth = {"$\\Omega_m$": 0.3, "$w\\ \\mathrm{Blinded}$": -1.0, "$\\Omega_\\Lambda$": 0.7}
            shift_params = truth if args.get("SHIFT") else None

            for index, (basename, outname) in enumerate(zip(args.get("INPUT_FILES"), args.get("PARSED_FILES"))):
                if args.get("NAMES"):
                    name = args.get("NAMES")[index].replace("_", " ")
                else:
                    name = os.path.basename(basename).replace("_", " ")

                # Do smarter biascor
                # eg name might be "(SN) ALL 5YR SCATTER"
                if ")" in name:
                    key = name.split(")", 1)[1]
                    # Key would now be " ALL 5YR SCATTER"
                    # Now also remove the COVOPT from the name
                    key = " ".join(key.split()[:-1])
                    # Key would now be " ALL 5YR"
                else:
                    key = name
                if key not in biases:
                    biases[key] = b
                    b += 1
                bias_index = biases[key]
                parse_chains(basename, outname, args, bias_index)

        logging.info("Finishing gracefully")
    except Exception as e:
        logging.exception(str(e))
        raise e
