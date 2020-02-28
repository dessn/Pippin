import numpy as np
import yaml
from chainconsumer import ChainConsumer
import pandas as pd
import sys
import argparse
import os
import logging


def fail(msg, condition=True):
    if condition:
        logging.error(msg)
        raise ValueError(msg)


def setup_logging():
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler("plot_cosmomc.log")])
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input yml file", type=str)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config.update(config["COSMOMC"])

    if config.get("NAMES") is not None:
        assert len(config["NAMES"]) == len(config["PARSED_FILES"]), (
            "You should specify one name per base file you pass in." + f" Have {len(config['PARSED_FILES'])} base names and {len(config['NAMES'])} names"
        )
    return config


def load_output(basename):
    if os.path.exists(basename):
        logging.warning(f"Loading in pre-saved CSV file from {basename}")
        df = pd.read_csv(basename)
        return df["_weight"].values, df["_likelihood"].values, df.iloc[:, 2:].to_numpy(), list(df.columns[2:])
    else:
        return None


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        if args.get("PARSED_FILES"):
            logging.info("Creating chain consumer object")
            c = ChainConsumer()
            do_full = False
            biases = {}
            b = 1
            truth = {"$\\Omega_m$": 0.3, "$w\\ \\mathrm{Blinded}$": -1.0, "$\\Omega_\\Lambda$": 0.7}
            shift_params = truth if args.get("SHIFT") else None

            for index, basename in enumerate(args.get("PARSED_FILES")):
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

                linestyle = "-" if name.lower().endswith("all") else "--"

                weights, likelihood, labels, chain, f = load_output(basename)
                if args.get("PRIOR"):
                    prior = args.get("PRIOR", 0.01)
                    logging.info(f"Applying prior width {prior} around 0.3")
                    om_index = labels.index("$\\Omega_m$")
                    from scipy.stats import norm

                    prior = norm.pdf(chain[:, om_index], loc=0.3, scale=prior)
                    weights *= prior

                do_full = do_full or f
                c.add_chain(chain, weights=weights, parameters=labels, name=name, posterior=likelihood, shift_params=shift_params, linestyle=linestyle)

            # Write all our glorious output
            out = args.get("OUTPUT_NAME")
            c.configure(plot_hists=False)
            c.analysis.get_latex_table(filename=out + "_params.txt")
            c.plotter.plot(filename=out + ".png", figsize=2.0)
            c.plotter.plot_summary(filename=out + "_summary.png", errorbar=True)
            if do_full:
                c.plotter.plot_walks(filename=out + "_walks.png")

        logging.info("Finishing gracefully")
    except Exception as e:
        logging.exception(str(e))
        raise e
