import numpy as np
import yaml
import pandas as pd
import sys
import argparse
import os
import logging


def setup_logging():
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[handler, logging.FileHandler("plot_errbudget.log")],
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def fail(msg, condition=True):
    if condition:
        logging.error(msg)
        raise ValueError(msg)


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input yml file", type=str)
    parser.add_argument(
        "-d", "--donefile", help="Path of done file", type=str, default="errbudget.done"
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config.update(config["COSMOMC"])

    if config.get("NAMES") is not None:
        assert len(config["NAMES"]) == len(config["PARSED_FILES"]), (
            "You should specify one name per base file you pass in."
            + f" Have {len(config['PARSED_FILES'])} base names and {len(config['NAMES'])} names"
        )
    return config


def load_output(basename):
    if os.path.exists(basename):
        logging.info(f"Loading in pre-saved CSV file from {basename}")
        df = pd.read_csv(basename)
        return (
            df["_weight"].values,
            df["_likelihood"].values,
            df.iloc[:, 2:].to_numpy(),
            list(df.columns[2:]),
        )
    else:
        fail(f"Cannot find file {basename}")
        return None


def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights, axis=0)
    variance = np.average((values - average) ** 2, weights=weights, axis=0)
    return average, np.sqrt(variance)


def get_entry(name, syst, filename):
    w, l, chain, cols = load_output(filename)
    means, stds = weighted_avg_and_std(chain, w)
    d = dict([(l + " avg", [x]) for l, x in zip(cols, means)])
    d.update(dict([(l + " std", [x]) for l, x in zip(cols, stds)]))
    d["name"] = [" ".join([n for n in name.split()[:-1]])]
    d["covopt"] = [syst]
    df = pd.DataFrame(d)
    return df


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()
    try:
        files = args.get("PARSED_FILES")
        names = args.get("NAMES")
        if files:
            logging.info("Making Error Budgets")

            budget_labels = [n.split()[-1] for n in names]
            bases = [
                (b, i)
                for i, b in enumerate(budget_labels)
                if b in ["NOSYS", "STAT", "STATONLY"]
            ]
            if len(bases):
                base, base_index = bases[0]
                data = [
                    get_entry(n, b, f) for n, b, f in zip(names, budget_labels, files)
                ]
                others = [d for i, d in enumerate(data) if i != base_index]

                base_df = data[base_index]
                df_all = pd.concat([base_df] + others).reset_index()

                df_all.columns = [
                    c.replace(r"\ \mathrm{Blinded}", "") for c in df_all.columns
                ]

                # Save out all the means + stds to file unchanged.
                df_all.to_csv(
                    "errbudget_all_uncertainties.csv", index=False, float_format="%0.4f"
                )

                # At this point, we have all the loaded in to a single dataframe, and now we group by name, compute the metrics, and save to file
                dfg = df_all.groupby("name")
                for name, df in dfg:
                    if "SN" not in name.upper():
                        continue
                    logging.info(f"Determining error budget for {name}")
                    output_filename = f"errbudget_{name}.txt".replace(" ", "_")

                    nosys_mask = df.covopt.str.upper().isin(
                        ["NOSYS", "NO_SYS", "STAT", "STATONLY", "STAT_ONLY"]
                    )
                    assert nosys_mask.sum() == 1, (
                        f"Multiple potential no systematic covopts found for name {name}, this is an issue"
                    )

                    avg_cols = [c for c in df.columns if c.endswith(" avg")]
                    std_cols = [c for c in df.columns if c.endswith(" std")]
                    delta_cols = [
                        c.replace(" avg", " delta")
                        for c in df.columns
                        if c.endswith(" avg")
                    ]
                    contrib_cols = [
                        c.replace(" std", " contrib")
                        for c in df.columns
                        if c.endswith(" std")
                    ]

                    df[delta_cols] = (
                        df.loc[:, avg_cols] - df.loc[nosys_mask, avg_cols].to_numpy()
                    )

                    min_var = (df.loc[nosys_mask, std_cols].to_numpy()) ** 2
                    df[contrib_cols] = np.sqrt(df.loc[:, std_cols] ** 2 - min_var)

                    df = df.reindex(sorted(df.columns)[::-1], axis=1)
                    df.to_latex(
                        output_filename, index=False, escape=False, float_format="%0.3f"
                    )
                    rep_file = output_filename.replace(".txt", "_repr.txt")

                    pd.set_option("display.max_rows", 500)
                    pd.set_option("display.max_columns", 500)
                    pd.set_option("display.width", 2000)

                    with open(rep_file, "w") as f:
                        f.write(df.reset_index(drop=True).__repr__())

    except Exception as e:
        logging.exception(e, exc_info=True)
        raise e
