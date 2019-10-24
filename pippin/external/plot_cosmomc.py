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
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler("plot_cosmomc.log")])
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
            logging.error(f"Cannot find blinding column {c} in list of names {names}")
            raise e


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input yml file", type=str)
    parser.add_argument("-d", "--donefile", help="Path of done file", type=str, default="plot_cosmomc.done")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config["donefile"] = args.donefile
    config.update(config["COSMOMC"])

    if config.get("NAMES") is not None:
        assert len(config["NAMES"]) == len(config["FILES"]), (
            "You should specify one name per base file you pass in." + f" Have {len(config['FILES'])} base names and {len(config['NAMES'])} names"
        )
    return config


def get_output_name(args, name):
    path = args["OUTPUT"] + "___" + name + ".csv.gz"
    basename = os.path.basename(path)
    return path, basename


def get_output(basename, args, index, name):
    output_path, b = get_output_name(args, name)
    res = load_output(b)
    if res is None:
        full = True
        logging.info("Loading in data from original CosmoMC files")
        param_file = os.path.join(basename) + ".paramnames"
        chain_files = get_chain_files(basename)
        names, labels = load_params(param_file)
        blind_params = args.get("BLIND")
        params = args.get("PARAMS")
        weights, likelihood, chain = load_chains(chain_files, names, args["params"])
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
        output_df.to_csv(output_path, float_format="%0.5f", index=False)
    else:
        full = False
        weights, likelihood, chain, labels = res
    logging.info(f"Chain for {basename} has shape {chain.shape}")
    logging.info(f"Labels for {basename} are {labels}")
    return weights, likelihood, labels, chain, full


def load_output(basename):
    if os.path.exists(basename):
        logging.warning("Loading in pre-saved CSV file. Be warned.")
        df = pd.read_csv(basename)
        return df["_weight"].values, df["_likelihood"].values, df.iloc[:, 2:].values, list(df.columns[2:])
    else:
        return None


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        if args.get("FILES"):
            logging.info("Creating chain consumer object")
            c = ChainConsumer()
            do_full = False
            biases = {}
            b = 1
            truth = {"$\\Omega_m$": 0.3, "$w\\ \\mathrm{Blinded}$": -1.0, "$\\Omega_\\Lambda$": 0.7}
            shift_params = truth if args.get("SHIFT") else None

            for index, basename in enumerate(args.get("FILES")):
                if args.get("NAMES"):
                    name = args.get("NAMES")[index].replace("_", " ")
                else:
                    name = os.path.basename(basename).replace("_", " ")
                # Do smarter biascor
                if ")" in name:
                    key = name.split(")", 1)[1]
                else:
                    key = name
                if key not in biases:
                    biases[key] = b
                    b += 1
                bias_index = biases[key]

                weights, likelihood, labels, chain, f = get_output(basename, args, bias_index, name)

                if args.get("PRIOR"):
                    prior = args.get("PRIOR", 0.01)
                    logging.info(f"Applying prior width {prior} around 0.3")
                    om_index = labels.index("$\\Omega_m$")
                    from scipy.stats import norm

                    prior = norm.pdf(chain[:, om_index], loc=0.3, scale=prior)
                    weights *= prior

                do_full = do_full or f
                c.add_chain(chain, weights=weights, parameters=labels, name=name, posterior=likelihood, shift_params=shift_params)

            # Write all our glorious output
            out = args.get("OUTPUT")
            c.analysis.get_latex_table(filename=out + "_params.txt")
            c.plotter.plot(filename=out + ".png", figsize=1.5)
            c.plotter.plot_summary(filename=out + "_summary.png", errorbar=True)
            if do_full:
                c.plotter.plot_walks(filename=out + "_walks.png")

        with open(args["donefile"], "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        logging.exception(str(e))
        with open(args["donefile"], "w") as f:
            f.write("FAILURE")
