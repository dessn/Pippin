import numpy as np
from chainconsumer import ChainConsumer
import pandas as pd
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
    steps = data.shape[0]
    data = [d.iloc[int(steps*0.3):, :] for d in data]

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
    logging.basicConfig(level=logging.INFO, format=fmt)


def blind(chain, names, columns_to_blind):
    np.random.seed(123)
    for i, c in enumerate(columns_to_blind):
        logging.info(f"Blinding column {c}")
        try:
            index = names.index(c)
            scale = np.random.normal(loc=1, scale=0.1, size=1000)[321 + i]
            chain[:, index] *= scale
        except ValueError as e:
            logging.error(f"Cannot find blinding column {c} in list of names {names}")
            raise e


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="The base to use for paramnames Eg /path/SN_CMB_OMW_ALL", nargs="+", type=str)
    parser.add_argument("-p", "--params", help="Param names to plot", nargs="*", type=str, default=["omegam", "w"])
    parser.add_argument("-n", "--name", help="Name of plot: eg name_without_extension", type=str, default="output")
    parser.add_argument("-b", "--blind", help="Blind these parameters", nargs="*", type=str, default=None)
    parser.add_argument("-d", "--donefile", help="Path of done file", type=str, default="done.txt")
    return parser.parse_args()


def get_output(basename, args):
    param_file = os.path.join(basename) + ".paramnames"
    chain_files = get_chain_files(basename)
    names, labels = load_params(param_file)
    weights, likelihood, chain = load_chains(chain_files, names, args.params)
    if args.blind:
        blind(chain, args.params or names, args.blind)
    labels = [f"${l}" + (r"\ \mathrm{Blinded}" if u in args.blind else "") + "$" for u in args.params for l, n in zip(labels, names) if n == u]
    logging.info(f"Chain for {basename} has shape {chain.shape}")
    logging.info(f"Labels for {basename} are {labels}")
    return weights, likelihood, labels, chain


if __name__ == "__main__":
    args = get_arguments()
    try:
        setup_logging()
        logging.info("Creating chain consumer object")
        c = ChainConsumer()
        for basename in args.basename:
            weights, likelihood, labels, chain = get_output(basename, args)
            name = os.path.basename(basename).replace("_", " ")
            c.add_chain(chain, weights=weights, parameters=labels, name=name, posterior=likelihood)

        # Write all our glorious output
        c.analysis.get_latex_table(filename=args.name + "_params.txt")
        c.plotter.plot(filename=args.name + ".png", figsize=1.5)
        c.plotter.plot_summary(filename=args.name + "_summary.png", errorbar=True)
        c.plotter.plot_walks(filename=args.name + "_walks.png")

        with open(args.donefile, "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        logging.error(str(e))
        with open(args.donefile, "w") as f:
            f.write("FAILURE")
