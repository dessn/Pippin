import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

field_names = ["SHALLOW", "DEEP"]
bands = ["g", "r", "i", "z"]
cuts = [30]


def print_drop(mask, name):
    print(f"Mask {name} rejects {100 * (~mask).sum() / mask.size:0.2f}% of samples")


def parse_data(filename):
    print(f"Parsing {filename}")
    filename_obs = filename.replace(".pkl", ".fitres")
    print(f"Reading FITRES dump file {filename_obs}")
    all_data = pd.read_csv(filename_obs, delim_whitespace=True, comment="#")

    # Generate extra columns
    all_data["ERRRATIO"] = all_data["FLUXCAL_DATA_ERR"] / all_data["FLUXERRCALC_SIM"]
    all_data["ERRTEST"] = all_data["FLUXERRCALC_SIM"] / all_data["FLUXCAL_DATA_ERR"]
    all_data["RATIO"] = all_data["FLUXCAL_DATA"] / all_data["SBFLUXCAL"]
    all_data["SBMAG"] = 27.5 - 2.5 * np.log10(all_data["SBFLUXCAL"])
    all_data["FLUXDIFF"] = all_data["FLUXCAL_DATA"] - all_data["FLUXCAL_SIM"]
    all_data["DEVIATION"] = all_data["FLUXDIFF"] / all_data["FLUXCAL_DATA_ERR"]
    all_data["DEVIATIONSIM"] = all_data["FLUXDIFF"] / all_data["FLUXERRCALC_SIM"]
    all_data["ABSDEVIATIONSIM"] = all_data["DEVIATIONSIM"].abs()
    all_data["ABSDEVIATION"] = all_data["DEVIATION"].abs()
    all_data["CHI2"] = all_data["FLUXCAL_DATA"] ** 2
    all_data["SNR"] = all_data["FLUXCAL_DATA"] / all_data["FLUXCAL_DATA_ERR"]
    neg_snr = all_data["SNR"] < 1
    all_data["SNRCLIP"] = all_data["SNR"]
    all_data.loc[neg_snr, "SNRCLIP"] = 1
    all_data["LOGSNR"] = np.log10(all_data["SNRCLIP"])

    # Get rid of initial rejects
    all_data = all_data.loc[all_data["REJECT"] == 0, :]

    # Determine impact of our masks and apply them
    mask_errtest_high = all_data["ERRTEST"] < 10
    mask_errtest_low = all_data["ERRTEST"] >= 0.1
    mask_sbfluxcal = all_data["SBFLUXCAL"] > 0
    mask_5yr = all_data["MJD"] < 57850
    mask_nolowz_y1 = (all_data["MJD"] > 56800) | (all_data["zHD"] > 0.13)
    # mask_fitprob = all_data["FITPROB"] > 0.01

    # mask_ratio = all_data["RATIO"] > 0
    mask = (
        mask_errtest_high
        & mask_errtest_low
        & mask_sbfluxcal
        & mask_5yr
        & mask_nolowz_y1
    )  # & mask_fitprob  # & mask_fluxcalsim & mask_ratio
    print_drop(mask_errtest_high, "errtest < 10")
    print_drop(mask_errtest_low, "errtest >= 0.1")
    print_drop(mask_sbfluxcal, "SBFLUXCAL > 0")
    print_drop(mask_5yr, "MJD < 57850")
    print_drop(mask_nolowz_y1, "No Y1 z<0.13")
    # print_drop(mask_fitprob, "fitprob >0.01")
    print_drop(mask, "Combined mask")

    all_data = all_data.loc[mask, :]

    # all_data = all_data.loc[:, ["CID", "MJD", "FIELD", "IFILTOBS", "LOGSNR", "SBMAG", "PSF", "ERRRATIO", "ERRTEST", "DEVIATION", "DEVIATIONSIM", "FLUXCAL_DATA", "FLUXCAL_SIM", "FLUXCAL_DATA_ERR", "FLUXERRCALC_SIM"]]
    print(f"Saving merged and thresholded data to {filename}")
    all_data.to_pickle(filename)


def get_data(filename, cut, cut_fitprob=0.0):
    df = pd.read_pickle(filename)
    mask2 = np.abs(df["DEVIATIONSIM"]) < cut
    print_drop(mask2, f"{cut} sigma outlier")
    mask_fitprob = df["FITPROB"] >= cut_fitprob
    print_drop(mask_fitprob, f"FITPROB >= {cut_fitprob}")
    return df.loc[mask2 & mask_fitprob, :]


def rejstd(x, debug=False):
    ns = 2
    d = 4
    mask = np.isfinite(x)
    x = x[mask]
    for i in range(ns):
        std = np.std(x)
        mean = np.mean(x)
        deviation = np.abs(x - mean) / std
        m = deviation < d
        x = x[m]

    s = np.std(x)
    n = x.size
    if n < 15:
        return np.NaN
    return s


def rejmean(x, debug=False):
    ns = 2
    d = 4
    mask = np.isfinite(x)
    x = x[mask]
    for i in range(ns):
        std = np.std(x)
        mean = np.mean(x)
        deviation = np.abs(x - mean) / std
        m = deviation < d
        x = x[m]

    s = np.mean(x)
    n = x.size
    if n < 15:
        return np.NaN
    return s


os.makedirs("maps", exist_ok=True)

if not os.path.exists("fakes_obs.pkl"):
    parse_data("fakes_obs.pkl")
    parse_data("sim_obs.pkl")
    print("Finished parsing")

for cut in cuts:
    for cut_fitprob in [0.00]:  # , 0.01]:
        print("Loading data")
        df_f = get_data("fakes_obs.pkl", cut, cut_fitprob=cut_fitprob)
        df_s = get_data("sim_obs.pkl", cut, cut_fitprob=cut_fitprob)

        data = [
            [
                ("LOGSNR", np.arange(0, 3.1, 1)),
                ("SBMAG", np.arange(20, 30, 2)),
                ("PSF", np.array([1, 3, 5])),
            ],
            [("LOGSNR", np.arange(0, 3.1, 1)), ("SBMAG", np.arange(20, 30, 1))],
            [("SBMAG", np.arange(20, 30, 1))],
        ]
        # data = [[("LOGSNR", np.arange(0, 3, 0.5)), ]]
        # data = [[("SBMAG", np.arange(20, 30, 1))]]

        # Digitize our data into the right bins
        print("Digitising")
        for maps in data:
            indices_f = []
            indices_s = []
            bcs = []
            shape = []
            for k, bins in maps:
                if bins is None:
                    bins = np.linspace(
                        df_f[k].quantile(0.001), df_f[k].quantile(0.999), 7
                    )
                bcs.append(0.5 * (bins[:-1] + bins[1:]))
                shape.append(bcs[-1].size)
                indices_f.append(np.digitize(df_f[k], bins=bins) - 1)
                indices_s.append(np.digitize(df_s[k], bins=bins) - 1)

            data = {}
            for field_name, color in zip(field_names, ("viridis", "magma")):
                fields_list = (
                    ["E1", "E2", "S1", "S2", "C1", "C2", "X1", "X2"]
                    if field_name == "SHALLOW"
                    else ["C3", "X3"]
                )

                for band_index, band in enumerate(bands):
                    print(f"Doing field {field_name} and band {band}")

                    # Select down to field and band
                    mask_f = (df_f["IFILTOBS"] == band_index + 2) & (
                        np.isin(df_f["FIELD"], fields_list)
                    )
                    mask_s = (df_s["IFILTOBS"] == band_index + 2) & (
                        np.isin(df_s["FIELD"], fields_list)
                    )
                    df_f2 = df_f[mask_f]
                    df_s2 = df_s[mask_s]
                    indices_f2 = [i[mask_f] for i in indices_f]
                    indices_s2 = [i[mask_s] for i in indices_s]

                    # Create empty arrays to store everything
                    print("Creating arrays")
                    numerator = np.empty(shape)
                    denom_rms = np.empty(shape)
                    denom_ratio = np.empty(shape)
                    numerator[:] = np.nan
                    denom_rms[:] = np.nan
                    denom_ratio[:] = np.nan

                    # For each element in the list, do the math
                    for index, x in np.ndenumerate(numerator):
                        # print(index)
                        mask_s = np.ones(df_s2.shape[0], dtype=bool)
                        mask_f = np.ones(df_f2.shape[0], dtype=bool)
                        for i, ind_f, ind_s in zip(index, indices_f2, indices_s2):
                            mask_s &= ind_s == i
                            mask_f &= ind_f == i
                        df_s3 = df_s2[mask_s]
                        df_f3 = df_f2[mask_f]

                        deviation_f = df_f3["DEVIATIONSIM"]
                        deviation_s = df_s3["DEVIATIONSIM"]
                        errratio = df_f3["ERRRATIO"]

                        numerator[index] = rejstd(deviation_f)
                        denom_rms[index] = rejstd(deviation_s)
                        denom_ratio[index] = rejmean(errratio)

                    # Construct final matrices
                    sims = numerator / denom_rms
                    fakes = numerator / denom_ratio

                    # Fill in missing values
                    print("Interpolating")
                    for i, (x, n) in enumerate([(sims, "SIM"), (fakes, "FAKES")]):
                        ind = np.indices(x.shape).reshape((len(x.shape), -1))
                        xx = x.flatten()
                        non_nan = np.isfinite(xx)

                        x2 = xx[non_nan]
                        ind2 = ind[:, non_nan]
                        x = (
                            griddata(ind2.T, x2, ind.T, method="nearest")
                            .T.flatten()
                            .reshape((x.shape))
                        )

                        # Save to output
                        data[n + field_name + band] = x

            # Create output files
            for j, n in enumerate(["SIM", "FAKES"]):
                output_string = []
                output_string.append(
                    "DEFINE_FIELDGROUP: SHALLOW E1+E2+S1+S2+C1+C2+X1+X2"
                )
                output_string.append("DEFINE_FIELDGROUP: DEEP C3+X3\n")
                names = [m[0] for m in maps]
                for field in field_names:
                    for b in bands:
                        d = data[n + field + b]

                        output_string.append("MAPNAME: FLUXERR_SCALE")
                        output_string.append(f"BAND: {b}  FIELD: {field}")
                        output_string.append(f"VARNAMES:  {' '.join(names)} ERRSCALE")
                        for inds in np.indices(d.shape).reshape((len(d.shape), -1)).T:
                            # Get the bin centers
                            bc = [f"{bc[i]:0.2f}" for bc, i in zip(bcs, inds)]
                            # Add on the data value
                            value = d[tuple(inds)]
                            bc.append(f"{value:0.3f}")
                            output_string.append("ROW: " + "  ".join(bc))
                        output_string.append("ENDMAP:\n")
                with open(
                    f"maps/DES5YR_{n}_ERRORFUDGES_DIFFIMG_{'_'.join(names)}.DAT", "w"
                ) as ff:
                    ff.write("\n".join(output_string))
