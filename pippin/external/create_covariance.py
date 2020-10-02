import argparse
import logging
import shutil

import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import yaml
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt

# TODO: Write out corr matrix separately to help debugging. Maybe make some plots as well for debugging. Also a readable txt file with corr.
# TODO: Add ordered list of systematics
# - delta mu vs z for systematics of chocie
# Note: choice should be determined automatically
# ie systematics that shift things more that like 0.2mag
# Text and plots
# Look for gradients in the diff vector over redshift to see what will impact cosmo
# output sorted list of gradients for each contribution (note be simple and use equal weighting for bins/sn? Or add an error floor)
# Make a dedicated CosmoMC folder to differentiate the human readable stuff

# TODO: Add analyse step to plot/copy corr
# TODO: Create readme analog / SUBMIT.INFO in top level output to explain wtf is going on (how all the files work together)
# TODO: Maybe put all useful output in a single YML file so we can easily put this into another fitter
# TODO: Add file which maps the COVOPT to the integers
# TODO: Get this working for each SN, not just each bin
# TODO: Check the size of the number of SN
# TODO: Get list of features that would be useful
# TODO: Make more explicit the checking for matched bins and number of supernova
# QUESTION: We write out zcmb = zhel, which does CosmoMC actually use? Should we not write out the actual values?
# TODO: Make CosmoMC starting point for Rick and Viv to debug and make a generic SN dataset f90 input

# TODO: Write out corr matrix separately to help debugging
# TODO: Add analyse step to plot corr
# TODO: Create readme analog / SUBMIT.INFO in top level output to explain wtf is going on (how all the files work together)
# TODO: Maybe put all useful output in a single YML file so we can easily put this into another fitter
# TODO: Add file which maps the COVOPT to the integers
# TODO: Get this working for each SN, not just each bin
# TODO: Get list of features that would be useful


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("seaborn").setLevel(logging.ERROR)


def read_yaml(path):
    logging.debug(f"Reading YAML from {path}")
    with open(path) as f:
        return yaml.safe_load(f.read())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="the name of the yml config file to run.")
    parser.add_argument("-u", "--unbinned", help="Utilise individual SN instead of binning", action="store_true")
    return parser.parse_args()


def load_data(path):
    if not os.path.exists(path):
        raise ValueError(f"Cannot load data from {path} - it doesnt exist")
    df = pd.read_csv(path, delim_whitespace=True, comment="#")
    logging.debug(f"\tLoaded data with shape {df.shape} from {path}")

    # Do a bit of data cleaning
    df = df.replace(999.0, np.inf)
    # M0DIF doesnt have MU column, so add it back in
    if "MU" not in df.columns:
        df["MU"] = df["MUREF"] + df["MUDIF"]

    # Sort to ensure direct subtraction comparison
    if "CID" in df.columns:
        df = df.sort_values(["zHD", "CID"])
    elif "z" in df.columns:
        df = df.sort_values("z")
    # The FITRES and M0DIF files have different column names
    if "CID" in df.columns:
        df = df.rename(columns={"MURES": "MUDIF", "MUERR": "MUDIFERR", "zHD": "z"})

    df = df.drop(columns=["VARNAMES:"])
    return df


def get_data_files(folder, individual):
    logging.debug(f"Loading all data files in {folder}")
    result = {}
    for file in os.listdir(folder):
        if (not individual and ".M0DIF" in file) or (individual and ".FITRES" in file and "MUOPT" in file):
            label = file.replace(".gz", "").replace(".M0DIF", "").replace(".FITRES", "")
            result[label] = load_data(folder / file)
    return result


def get_fitopt_muopt_from_name(name):
    f = int(name.split("FITOPT")[1][:3])
    m = int(name.split("MUOPT")[1][:3])
    return f, m


def get_name_from_fitopt_muopt(f, m):
    return f"FITOPT{f:03d}_MUOPT{m:03d}"


def get_fitopt_scales(lcfit_info, sys_scales):
    """ Returns a dict mapping FITOPT numbers to (label, scale) """
    fitopt_list = lcfit_info["FITOPT_LIST"]
    result = {}
    for name, label, _ in fitopt_list:
        if label != "DEFAULT":
            if label not in sys_scales:
                logging.warning(f"No FITOPT scale found for {label}")
        scale = sys_scales.get(label, 1.0)
        d = int(name.replace("FITOPT", ""))
        result[d] = (label, scale)
    return result


def get_cov_from_diff(df1, df2, scale):
    """ Returns both the covariance contribution and summary stats (slope and mean abs diff) """
    diff = scale * (df1["MUDIF"].to_numpy() - df2["MUDIF"].to_numpy())
    cov = diff[:, None] @ diff[None, :]

    # Determine the gradient using simple linear regression
    reg = LinearRegression()
    weights = 1 / np.sqrt(0.003 ** 2 + df1["MUDIFERR"] ** 2 + df2["MUDIFERR"] ** 2)
    reg.fit(df1[["z"]], diff, sample_weight=weights)
    coef = reg.coef_[0]

    mean_abs_deviation = np.average(np.abs(diff), weights=weights)
    max_abs_deviation = np.max(np.abs(diff))
    return cov, (coef, mean_abs_deviation, max_abs_deviation)


def get_contributions(m0difs, fitopt_scales, muopt_labels):
    """ Gets a dict mapping 'FITOPT_LABEL|MUOPT_LABEL' to covariance)"""
    result, slopes = {}, []
    base = None

    for name, df in m0difs.items():
        f, m = get_fitopt_muopt_from_name(name)
        logging.debug(f"Determining contribution for FITOPT {f}, MUOPT {m}")

        # Get label and scale for FITOPTS and MUOPTS. Note 0 index is DEFAULT
        if f == 0:
            fitopt_label = "DEFAULT"
            scale = 1.0
        else:
            fitopt_label, scale = fitopt_scales[f]
        muopt_label = muopt_labels[m] if m else "DEFAULT"

        # Depending on f and m, compute the contribution to the covariance matrix
        if f == 0 and m == 0:
            # This is the base file, so don't return anything. CosmoMC will add the diag terms itself.
            cov = np.zeros((df["MU"].size, df["MU"].size))
            summary = 0, 0, 0
            base = df
        elif m > 0:
            # This is a muopt, to compare it against the MUOPT000 for the same FITOPT
            df_compare = m0difs[get_name_from_fitopt_muopt(f, 0)]
            cov, summary = get_cov_from_diff(df, df_compare, scale)
        else:
            # This is a fitopt with MUOPT000, compare to base file
            df_compare = m0difs[get_name_from_fitopt_muopt(0, 0)]
            cov, summary = get_cov_from_diff(df, df_compare, scale)

        result[f"{fitopt_label}|{muopt_label}"] = cov
        slopes.append([name, fitopt_label, muopt_label, *summary])

    summary_df = pd.DataFrame(slopes, columns=["name", "fitopt_label", "muopt_label", "slope", "mean_abs_deviation", "max_abs_deviation"])
    summary_df = summary_df.sort_values(["slope", "mean_abs_deviation", "max_abs_deviation"], ascending=False)
    return result, base, summary_df


def apply_filter(string, pattern):
    """ Used for matching COVOPTs to FITOPTs and MUOPTs"""
    if pattern.startswith("="):
        return string == pattern[1:]
    elif pattern.startswith("+"):
        return pattern[1:] in string
    elif pattern.startswith("-"):
        return pattern[1:] not in string
    elif pattern == "":
        return True
    else:
        raise ValueError(f"Unable to parse COVOPT matching pattern {pattern}")


def get_cov_from_covopt(covopt, contributions, base):
    # Covopts will come in looking like "[cal] [+cal,=DEFAULT]"
    # We have to parse this. Eventually can make this structured and move away from
    # legacy, but dont want to make too many people change how they are doing things
    # in one go
    label, fitopt_filter, muopt_filter = re.findall(r"\[(.*)\] \[(.*),(.*)\]", covopt)[0]
    logging.debug(f"Computing cov for COVOPT {label} with FITOPT filter '{fitopt_filter}' and MUOPT filter '{muopt_filter}'")

    final_cov = None

    for key, cov in contributions.items():
        fitopt_label, muopt_label = key.split("|")
        if apply_filter(fitopt_label, fitopt_filter) and apply_filter(muopt_label, muopt_filter):
            if final_cov is None:
                final_cov = cov.copy()
            else:
                final_cov += cov

    # Validate that the final_cov is invertible
    try:
        # CosmoMC will add the diag terms, so lets do it here and make sure its all good
        effective_cov = final_cov + np.diag(base["MUDIFERR"] ** 2)
        np.linalg.inv(effective_cov)
    except np.linalg.LinAlgError as ex:
        logging.exception(f"Unable to invert covariance matrix for COVOPT {label}")
        raise ex

    return label, final_cov


def write_dataset(path, data_file, cov_file, template_path):
    with open(template_path) as f_in:
        with open(path, "w") as f_out:
            f_out.write(f_in.read().format(data_file=data_file, cov_file=cov_file))


def write_data(path, base):
    zs = base["z"].to_numpy()
    mu = base["MU"].to_numpy()
    mbs = -19.36 + mu
    mbes = base["MUDIFERR"].to_numpy()

    # I am so sorry about this, but CosmoMC is very particular
    logging.info(f"Writing out data to {path}")
    with open(path, "w") as f:
        f.write("#name zcmb    zhel    dz mb        dmb     x1 dx1 color dcolor 3rdvar d3rdvar cov_m_s cov_m_c cov_s_c set ra dec biascor\n")
        for i, (z, mb, mbe) in enumerate(zip(zs, mbs, mbes)):
            f.write(f"{i:5d} {z:6.5f} {z:6.5f} 0  {mb:8.5f} {mbe:8.5f} 0 0 0 0 0 0 0 0 0 0 0 0\n")


def write_covariance(path, cov):
    logging.info(f"Writing covariance to {path}")

    # Write out the slopes
    with open(path, "w") as f:
        f.write(f"{cov.shape[0]}\n")
        for c in cov.flatten():
            f.write(f"{c:0.8f}\n")


def write_cosmomc_output(config, covs, base):
    # Copy INI files. Create covariance matrices. Create .dataset. Modifying INI files to point to resources
    out = Path(config["OUTDIR"]) / "cosmomc"
    data_file = out / f"data.txt"
    dataset_template = Path(config["COSMOMC_TEMPLATES"]) / config["DATASET_FILE"]
    dataset_files = []
    os.makedirs(out, exist_ok=True)

    # Create lcparam file
    write_data(data_file, base)

    # Create covariance matrices and datasets
    for i, (label, cov) in enumerate(covs):
        cov_file = out / f"sys_{i}.txt"
        dataset_file = out / f"dataset_{i}.txt"

        write_covariance(cov_file, cov)
        write_dataset(dataset_file, data_file, cov_file, dataset_template)
        dataset_files.append(dataset_file)

    # Copy some INI files
    ini_files = [f for f in os.listdir(config["COSMOMC_TEMPLATES"]) if f.endswith(".ini") or f.endswith(".yml") or f.endswith(".md")]
    for ini in ini_files:
        op = Path(config["COSMOMC_TEMPLATES"]) / ini

        if ini in ["base.ini"]:
            # If its the base.ini, just copy it
            npath = out / ini
            shutil.copy(op, npath)
        else:
            # Else we need one of each ini per covopt
            for i, (label, cov) in enumerate(covs):
                # Copy with new index
                npath = out / ini.replace(".ini", f"_{i}.ini")
                shutil.copy(op, npath)

                basename = os.path.basename(npath).replace(".ini", "")

                # Append the dataset info
                with open(npath, "a+") as f:
                    f.write(f"\nfile_root={basename}\n")
                    f.write(f"jla_dataset={dataset_files[i]}\n")


def write_summary_output(config, covariances, base):
    out = Path(config["OUTDIR"])
    info = {}
    cov_info = {}
    for i, (label, cov) in enumerate(covariances):
        cov_info[i] = label
    info["COVOPTS"] = cov_info

    logging.info("Writing INFO.YML")
    with open(out / "INFO.YML", "w") as f:
        yaml.safe_dump(info, f)


def write_correlation(path, label, cov, base):
    logging.debug(f"\tWriting out cov for COVOPT {label}")
    diag = np.sqrt(np.diag(cov))
    corr = cov / (diag[:, None] @ diag[None, :])
    np.fill_diagonal(corr, 1.0)
    np.savetxt(path, corr, fmt="%5.2f")
    corr = pd.DataFrame((corr * 100).astype(int), columns=base["z"], index=base["z"])

    precision = pd.DataFrame(np.linalg.inv(cov), columns=base["z"], index=base["z"])

    if corr.shape[0] < 100:
        height = 5 + 5  # corr.shape[0] // 4
        fig, axes = plt.subplots(figsize=(2 * height + 2, height), ncols=2)
        sb.heatmap(precision, annot=False, ax=axes[0], cmap="magma", square=True)
        sb.heatmap(corr, annot=True, fmt="d", ax=axes[1], cmap="RdBu", vmin=-100, vmax=100, square=True)
        axes[0].set_title("Precision matrix")
        axes[1].set_title("Correlation matrix (percent)")
        plt.tight_layout()
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=100)
    else:
        logging.info("\tMatrix is large, skipping plotting")


def write_debug_output(config, covariances, base, summary):
    # Plot correlation matrix
    out = Path(config["OUTDIR"])

    # The slopes can be used to figure out what systematics have largest impact on cosmology
    logging.info("Writing out summary.csv information")
    with open(out / "summary.csv", "w") as f:
        with pd.option_context("display.max_rows", 100000, "display.max_columns", 100, "display.width", 1000):
            f.write(summary.__repr__())

    logging.info("Showing correlation matrices:")
    diag = np.diag(base["MUDIFERR"] ** 2)
    for i, (label, cov) in enumerate(covariances):
        write_correlation(out / f"corr_{i}_{label}.txt", label, cov + diag, base)


def create_covariance(config, args):
    # Define all our pathing to be super obvious about where it all is
    input_dir = Path(config["INPUT_DIR"])
    data_dir = input_dir / config["VERSION"]
    sys_file = Path(config["SYSFILE"])

    # Read in all the needed data
    submit_info = read_yaml(input_dir / "SUBMIT.INFO")
    data = get_data_files(data_dir, args.unbinned)
    sys_scale = read_yaml(sys_file)

    # Also need to get the FITOPT labels from the original LCFIT directory
    lcfit_info = read_yaml(Path(submit_info["INPDIR_LIST"][0]) / "SUBMIT.INFO")
    fitopt_scales = get_fitopt_scales(lcfit_info, sys_scale)
    muopt_labels = {int(x.replace("MUOPT", "")): l for x, l, _ in submit_info["MUOPT_LIST"]}

    # Now that we have the data, figure out how each much each FITOPT/MUOPT pair contributes to cov
    contributions, base, summary = get_contributions(data, fitopt_scales, muopt_labels)

    # For each COVOPT, we want to find the contributions which match to construct covs for each COVOPT
    logging.info("Computing covariance for COVOPTS")
    covopts = ["[ALL] [,]"] + config["COVOPTS"]  # Adds the covopt to compute everything
    covariances = [get_cov_from_covopt(c, contributions, base) for c in covopts]

    write_cosmomc_output(config, covariances, base)
    write_summary_output(config, covariances, base)
    write_debug_output(config, covariances, base, summary)


if __name__ == "__main__":
    try:
        setup_logging()
        args = get_args()
        config = read_yaml(args.input)
        create_covariance(config, args)
    except Exception as e:
        logging.exception(e)
        raise e
