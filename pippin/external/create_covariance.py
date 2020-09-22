import argparse
import logging
import shutil

import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import yaml


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s")


def read_yaml(path):
    logging.debug(f"Reading YAML from {path}")
    with open(path) as f:
        return yaml.safe_load(f.read())


def get_args():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="the name of the yml config file to run.")
    return parser.parse_args()


def load_m0dif(path):
    if not os.path.exists(path):
        raise ValueError(f"Cannot load M0DIF from {path} - it doesnt exist")
    df = pd.read_csv(path, delim_whitespace=True, comment="#")
    logging.debug(f"\tLoaded M0DIF with shape {df.shape} from {path}")

    # Do a bit of data cleaning
    df = df.replace(999.0, np.inf)
    df["MU"] = df["MUREF"] + df["MUDIF"]
    df = df.drop(columns=["VARNAMES:"])
    return df


def get_m0difs(folder):
    logging.debug(f"Loading all M0DIF files in {folder}")
    result = {}
    for file in os.listdir(folder):
        if ".M0DIF" in file:
            label = file.replace(".gz", "").replace(".M0DIF", "")
            result[label] = load_m0dif(folder / file)
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
                logging.warning(f"No FITOPT scale found for {label} in input options: {sys_scales}")
        scale = sys_scales.get(label, 1.0)
        d = int(name.replace("FITOPT", ""))
        result[d] = (label, scale)
    return result


def get_cov_from_diff(df1, df2, scale):
    diff = scale * (df1["MUDIF"].to_numpy() - df2["MUDIF"].to_numpy())
    return diff[:, None] @ diff[None, :]


def get_contributions(m0difs, fitopt_scales, muopt_labels):
    """ Gets a dict mapping 'FITOPT_LABEL|MUOPT_LABEL' to covariance)"""
    result = {}
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
            cov = np.zeros((df["MUDIFERR"].size, df["MUDIFERR"].size))
            base = df
        elif m:
            # This is a muopt, to compare it against the MUOPT000 for the same FITOPT
            df_compare = m0difs[get_name_from_fitopt_muopt(f, 0)]
            cov = get_cov_from_diff(df, df_compare, scale)
        else:
            # This is a fitopt with MUOPT000, compare to base file
            df_compare = m0difs[get_name_from_fitopt_muopt(0, 0)]
            cov = get_cov_from_diff(df, df_compare, scale)

        result[f"{fitopt_label}|{muopt_label}"] = cov
    return result, base


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


def get_cov_from_covopt(covopt, contributions):
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
    return final_cov


def write_dataset(path, lcparam_file, cov_file):
    template = f"""name = JLA
data_file = {lcparam_file}
pecz = 0
intrinsicdisp = 0
twoscriptmfit = F
scriptmcut = 10.0
has_mag_covmat = T
mag_covmat_file =  {cov_file}
has_stretch_covmat = F
has_colour_covmat = F
has_mag_stretch_covmat = F
has_mag_colour_covmat = F
has_stretch_colour_covmat = F
"""
    with open(path, "w") as f:
        f.write(template)


def write_lcparam(path, base):
    zs = base["z"].to_numpy()
    mu = base["MU"].to_numpy()
    mbs = -19.36 + mu
    mbes = base["MUDIFERR"].to_numpy()

    # I am so sorry about this, but CosmoMC is very particular
    logging.info(f"Writing out lcparam data to {path}")
    with open(path, "w") as f:
        f.write("#name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar cov_m_s cov_m_c cov_s_c set ra dec biascor \n")
        for i, (z, mb, mbe) in enumerate(zip(zs, mbs, mbes)):
            f.write(f"{i} {z} {z} 0 {mb} {mbe} 0 0 0 0 0 0 0 0 0 0 0 0\n")


def write_covariance(path, cov):
    logging.info(f"Writing covariance to {path}")
    with open(path, "w") as f:
        f.write(f"{cov.shape[0]}\n")
        for c in cov.flatten():
            f.write(f"{c:0.8f}\n")


def write_output(config, covs, base):
    # Copy INI files. Create covariance matrices. Create .dataset. Modifying INI files to point to resources
    out = Path(config["OUTDIR"])
    lcparam_file = out / f"lcparam.txt"
    dataset_files = []

    # Create lcparam file
    write_lcparam(lcparam_file, base)

    # Create covariance matrices and datasets
    for i, cov in enumerate(covs):
        cov_file = out / f"sys_{i}.txt"
        dataset_file = out / f"dataset_{i}.txt"

        write_covariance(cov_file, cov)
        write_dataset(dataset_file, lcparam_file, cov_file)
        dataset_files.append(cov_file)

    # Copy some INI files
    ini_files = [f for f in os.listdir(config["COSMOMC_TEMPLATES"]) if f.endswith(".ini")]
    for ini in ini_files:
        op = Path(config["COSMOMC_TEMPLATES"]) / ini

        if ini in ["base.ini"]:
            # If its the base.ini, just copy it
            npath = out / ini
            shutil.copy(op, npath)
        else:
            # Else we need one of each ini per covopt
            for i, cov in enumerate(covs):
                # Copy with new index
                npath = out / ini.replace(".ini", f"_{i}.ini")
                shutil.copy(op, npath)

                basename = os.path.basename(npath).replace(".ini", "")

                # Append the dataset info
                with open(npath, "a+") as f:
                    f.write(f"\nfile_root={basename}\n")
                    f.write(f"jla_dataset={dataset_files[i]}\n")
                    f.write("root_dir = {root_dir}\n")


def create_covariance(config):
    # Define all our pathing to be super obvious about where it all is
    input_dir = Path(config["INPUT_DIR"])
    data_dir = input_dir / config["VERSION"]
    sys_file = Path(config["SYSFILE"])

    # Read in all the needed data
    submit_info = read_yaml(input_dir / "SUBMIT.INFO")
    m0difs = get_m0difs(data_dir)
    sys_scale = read_yaml(sys_file)

    # Also need to get the FITOPT labels from the original LCFIT directory
    lcfit_info = read_yaml(Path(submit_info["INPDIR_LIST"][0]) / "SUBMIT.INFO")
    fitopt_scales = get_fitopt_scales(lcfit_info, sys_scale)
    muopt_labels = {int(x.replace("MUOPT", "")): l for x, l, _ in submit_info["MUOPT_LIST"]}

    # Now that we have the data, figure out how each much each FITOPT/MUOPT pair contributes to cov
    contributions, base = get_contributions(m0difs, fitopt_scales, muopt_labels)

    # For each COVOPT, we want to find the contributions which match to construct covs for each COVOPT
    logging.info("Computing covariance for COVOPTS")
    covopts = ["[ALL] [,]"] + config["COVOPTS"]  # Adds the covopt to compute everything
    covariances = [get_cov_from_covopt(c, contributions) for c in covopts]

    write_output(config, covariances, base)


if __name__ == "__main__":
    try:
        setup_logging()
        args = get_args()
        config = read_yaml(args.input)
        create_covariance(config)
    except Exception as e:
        logging.exception(e, exc_info=e)
        raise e
