import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, binned_statistic, moment, norm, binned_statistic
from scipy.interpolate import interp2d, griddata
import matplotlib


field_names = ["SHALLOW", "DEEP"]
bands = ["g", "r", "i", "z"]
cuts = [4, 5, 7, 10]


def print_drop(mask, name):
    print(f"Mask {name} rejects {100*(~mask).sum()/mask.size:0.2f}% of samples")


def get_data(filename):
    all_data = pd.read_csv(filename, delim_whitespace=True, comment="#")

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
    # mask_ratio = all_data["RATIO"] > 0
    mask = mask_errtest_high & mask_errtest_low & mask_sbfluxcal & mask_5yr  # & mask_fluxcalsim & mask_ratio
    print_drop(mask_errtest_high, "errtest < 10")
    print_drop(mask_errtest_low, "errtest >= 0.1")
    print_drop(mask_sbfluxcal, "SBFLUXCAL > 0")
    print_drop(mask_5yr, "MJD < 57850")
    print_drop(mask, "Combined mask")

    all_data = all_data.loc[mask, :]
    # Finally, lets get rid of massive outlier points. 15sigma should be good enough.
    mask2 = all_data["ABSDEVIATIONSIM"] < cut
    print_drop(mask2, f"{cut} sigma outlier")
    return all_data.loc[mask2, :]


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


def get_stats(df, indicies, index1, index2, fields, band, debug=False):
    mask = (index1 == indicies[0, :]) & (index2 == indicies[1, :])
    deviation = df.loc[mask, "DEVIATIONSIM"]

    numerator = rejstd(deviation, debug=debug)
    if numerator < 0.7 or numerator > 5.0:
        print(
            f"std {numerator} with {df.shape[0]} size for band {band} and fields {fields}  \n   get_stats(df, indicies, {index1}, {index2}, {band}, '{fields}', debug=True)"
        )
    return numerator


def err_ratio(df, indicies, index1, index2, fields, band, debug=False):
    mask = (index1 == indicies[0, :]) & (index2 == indicies[1, :])
    ratio = df.loc[mask, "ERRRATIO"]
    numerator = rejmean(ratio, debug=debug)
    return numerator


filename_fakes = "fakes_obs.fitres.gz"
filename_sim = "sim_obs.fitres.gz"


for cut in cuts:
    run = True

    if run:
        # filename_fakes = "base.fitres"
        df_f = get_data(filename_fakes)
        df_s = get_data(filename_sim)
        df_f.sort_values(by="ABSDEVIATIONSIM", inplace=True, ascending=False)
        df_s.sort_values(by="ABSDEVIATIONSIM", inplace=True, ascending=False)

    data = [("LOGSNR", np.arange(0, 3, 0.5), "SBMAG", np.arange(20, 30, 1), "PSF", np.arange(0.5, 5.6, 1.0))]
    # data = [("LOGSNR", np.arange(0, 6, 0.5), "PSF", np.arange(1, 4.5, 0.5))]
    # data = [("SBMAG", np.arange(21, 30, 1.0), "PSF", np.arange(1, 4.5, 0.5))]

    for k1, bins1, k2, bins2, k3, bins3 in data:
        df1 = df_f[k1]
        df2 = df_f[k2]
        ds1 = df_s[k1]
        ds2 = df_s[k2]
        df3 = df_f[k3]
        ds3 = df_s[k3]
        if bins1 is None:
            bins1 = np.linspace(df1.quantile(0.001), df1.quantile(0.999), 15)
        bincs1 = 0.5 * (bins1[1:] + bins1[:-1])
        if bins2 is None:
            bins2 = np.linspace(df2.quantile(0.001), df2.quantile(0.999), 15)
        bincs2 = 0.5 * (bins2[1:] + bins2[:-1])
        if bins3 is None:
            bins3 = np.linspace(df3.quantile(0.001), df3.quantile(0.999), 15)
        bincs3 = 0.5 * (bins3[1:] + bins3[:-1])

        b1s = np.array([b1 for b1 in bincs1 for b2 in bincs2])
        b2s = np.array([b2 for b1 in bincs1 for b2 in bincs2])

        if run:
            cf, _, _, indicies_f = binned_statistic_2d(df1, df2, df1, statistic="count", bins=[bins1, bins2], expand_binnumbers=True)
            cs, _, _, indicies_s = binned_statistic_2d(ds1, ds2, ds1, statistic="count", bins=[bins1, bins2], expand_binnumbers=True)
            indicies_f -= 1
            indicies_s -= 1

        if True:
            plt.figure()

            shallow_bands = [{}, {}]
            deep_bands = [{}, {}]

            deep_psf = [{}, {}]
            shallow_psf = [{}, {}]

            for field_name, color in zip(field_names, ("viridis", "magma")):
                fig, axes = plt.subplots(nrows=6, ncols=4, sharex=True, figsize=(14, 14), gridspec_kw={"hspace": 0.2, "wspace": 0.25})

                fields_list = ["E1", "E2", "S1", "S2", "C1", "C2", "X1", "X2"] if field_name == "SHALLOW" else ["C3", "X3"]

                for band_index, (col, band) in enumerate(zip(axes.T, bands)):
                    print(f"Doing field {field_name} and band {band}")

                    mask_f = (df_f["IFILTOBS"] == band_index + 2) & (np.isin(df_f["FIELD"], fields_list))
                    mask_s = (df_s["IFILTOBS"] == band_index + 2) & (np.isin(df_s["FIELD"], fields_list))
                    df_f2 = df_f[mask_f]
                    df_s2 = df_s[mask_s]
                    indicies_f2 = indicies_f[:, mask_f]
                    indicies_s2 = indicies_s[:, mask_s]
                    print("Getting numerator")
                    numerators = np.array(
                        [get_stats(df_f2, indicies_f2, i1, i2, field_name, band) for i1, _ in enumerate(bincs1) for i2, _ in enumerate(bincs2)]
                    )
                    print("Getting denominator")
                    denominators = np.array(
                        [get_stats(df_s2, indicies_s2, i1, i2, field_name, band) for i1, _ in enumerate(bincs1) for i2, _ in enumerate(bincs2)]
                    )
                    print("Getting sim (eq 14) ratios")
                    ratios = numerators / denominators

                    print("Getting eq15 denominator")
                    eq_15_denom = np.array(
                        [err_ratio(df_f2, indicies_f2, i1, i2, field_name, band) for i1, _ in enumerate(bincs1) for i2, _ in enumerate(bincs2)]
                    )
                    print("Getting fakes (eq 15) ratios")
                    ratios_fakes = numerators / eq_15_denom

                    print("CCCC ", ratios_fakes, numerators, eq_15_denom)

                    # loop twice, once for ratios and one for ratios_fakes
                    for j, r in enumerate([ratios, ratios_fakes]):
                        print("In loop " + str(j))
                        shape = (bincs1.size, bincs2.size)
                        non_nan = np.isfinite(r)
                        ratios2 = griddata(np.vstack((b1s[non_nan], b2s[non_nan])).T, r[non_nan], np.vstack((b1s, b2s)).T, method="nearest").T.flatten()

                        # For the data points that satisfy the field and band, get the prediced ratio
                        # df_f2 = df_f[(df_f["IFILTOBS"] == band) & (np.isin(df_f["FIELD"], fields_list))]

                        ratios_s_lin = griddata(np.vstack((b1s, b2s)).T, ratios2, np.vstack((df_s2["LOGSNR"], df_s2["SBMAG"])).T, method="linear").T.flatten()
                        ratios_s_near = griddata(np.vstack((b1s, b2s)).T, ratios2, np.vstack((df_s2["LOGSNR"], df_s2["SBMAG"])).T, method="nearest").T.flatten()
                        ratios_s_lin[np.isnan(ratios_s_lin)] = ratios_s_near[np.isnan(ratios_s_lin)]

                        ratios_f_lin = griddata(np.vstack((b1s, b2s)).T, ratios2, np.vstack((df_f2["LOGSNR"], df_f2["SBMAG"])).T, method="linear").T.flatten()
                        ratios_f_near = griddata(np.vstack((b1s, b2s)).T, ratios2, np.vstack((df_f2["LOGSNR"], df_f2["SBMAG"])).T, method="nearest").T.flatten()
                        ratios_f_lin[np.isnan(ratios_f_lin)] = ratios_f_near[np.isnan(ratios_f_lin)]

                        # Compute the ratios
                        indices_s2 = np.digitize(df_s2["PSF"], bins3) - 1
                        indices_f2 = np.digitize(df_f2["PSF"], bins3) - 1
                        numerator = np.array([rejstd(df_f2.loc[indices_f2 == ii, "DEVIATIONSIM"]) for ii, _ in enumerate(bincs3)])
                        if j == 0:
                            denominator = np.array([rejstd(df_s2.loc[indices_s2 == ii, "DEVIATIONSIM"]) for ii, _ in enumerate(bincs3)])
                            ratio_means, _, _ = binned_statistic(df_s2["PSF"], ratios_s_lin, bins=bins3)

                        else:
                            denominator = np.array([rejmean(df_f2.loc[indices_f2 == ii, "ERRRATIO"]) for ii, _ in enumerate(bincs3)])
                            ratio_means, _, _ = binned_statistic(df_f2["PSF"], ratios_f_lin, bins=bins3)

                        ratio_psf = numerator / denominator
                        unexplained = ratio_psf / ratio_means
                        if j == 1:
                            print("DDDDDDD ", field_name, band, denominator, numerator, ratio_psf, ratio_means)

                        print(field_name, band, unexplained)
                        print("PSF: ", ratio_psf)

                        if field_name == "SHALLOW":
                            d = shallow_bands[j]
                            dp = shallow_psf[j]
                        else:
                            d = deep_bands[j]
                            dp = deep_psf[j]
                        if d.get(band) is None:
                            d[band] = []
                        if dp.get(band) is None:
                            dp[band] = []
                        for b1, b2, rr in zip(b1s, b2s, ratios2):
                            if np.isfinite(rr):
                                d[band].append(f"ROW:       {b1:0.2f}     {b2:0.2f}   {rr:0.3f}")
                        for bc, psf in zip(bincs3, unexplained):
                            if np.isfinite(psf):
                                dp[band].append(f"ROW:       {bc:0.2f}     {psf:0.3f}")

                        if j == 0:
                            *_, cc = col[0].hist2d(b1s, b2s, bins=[bins1, bins2], weights=ratios2, label=field_name, cmap=color, vmin=0)
                            cbar1 = fig.colorbar(cc, ax=col[0])
                            *_, cc = col[1].hist2d(b1s, b2s, bins=[bins1, bins2], weights=r, label=field_name, cmap=color, vmin=0)
                            cbar1 = fig.colorbar(cc, ax=col[1])
                            *_, cc = col[2].hist2d(b1s, b2s, bins=[bins1, bins2], weights=numerators, label=field_name, cmap=color, vmin=0)
                            cbar1 = fig.colorbar(cc, ax=col[2])
                            *_, cc = col[3].hist2d(b1s, b2s, bins=[bins1, bins2], weights=denominators, label=field_name, cmap=color, vmin=0)
                            cbar1 = fig.colorbar(cc, ax=col[3])
                            col[0].set_title(band)
                        elif j == 1:
                            *_, cc = col[4].hist2d(b1s, b2s, bins=[bins1, bins2], weights=ratios2, label=field_name, cmap=color, vmin=0)
                            cbar1 = fig.colorbar(cc, ax=col[4])
                            *_, cc = col[5].hist2d(b1s, b2s, bins=[bins1, bins2], weights=eq_15_denom, label=field_name, cmap=color, vmin=0)
                            cbar1 = fig.colorbar(cc, ax=col[5])

                axes[0][0].set_title("Sims - ratio interpolated")
                axes[1][0].set_title("Sims - ratio")
                axes[2][0].set_title("Sims - numerator")
                axes[3][0].set_title("Sims - denominator")
                axes[4][0].set_title("Fakes - ratio interpolated")
                axes[5][0].set_title("Fakes - denominator")
                axes[5][0].set_xlabel(k1)
                axes[5][1].set_xlabel(k1)
                axes[5][2].set_xlabel(k1)
                axes[5][3].set_xlabel(k1)
                axes[0][0].set_ylabel(k2)
                axes[1][0].set_ylabel(k2)
                axes[2][0].set_ylabel(k2)
                axes[3][0].set_ylabel(k2)

                plt.savefig(f"DES5YR_SIM_ERRORFUDGES_{k1}_{k2}_{field_name}_{cut}.png", dpi=150, bbox_inches="tight", transparent=True)
                plt.show()

        for j, n in enumerate(["SIM", "FAKES"]):
            output_string = []
            output_string.append("DEFINE_FIELDGROUP: SHALLOW E1+E2+S1+S2+C1+C2+X1+X2")
            output_string.append("DEFINE_FIELDGROUP: DEEP C3+X3")
            output_string.append("")
            output_string_pdf = []
            for f in field_names:
                for b in bands:
                    if field_name == "SHALLOW":
                        d = shallow_bands[j]
                        dp = shallow_psf[j]
                    else:
                        d = deep_bands[j]
                        dp = deep_psf[j]
                    output_string.append("")
                    output_string.append("MAPNAME: FLUXERR_SCALE_HSB")
                    output_string.append(f"BAND: {b} FIELD: {f}")
                    output_string.append("VARNAMES:  LOGSNR  SBMAG ERRSCALE")
                    output_string += d[b]
                    output_string.append("ENDMAP:\n")

                    output_string_pdf.append("")
                    output_string_pdf.append("MAPNAME: FLUXERR_SCALE_PSF")
                    output_string_pdf.append(f"BAND: {b} FIELD: {f}")
                    output_string_pdf.append("VARNAMES:  PSF  ERRSCALE")
                    output_string_pdf += dp[b]
                    output_string_pdf.append("ENDMAP:\n")

            with open(f"DES5YR_{n}_ERRORFUDGES_{k1}_{k2}_{cut}.DAT", "w") as ff:
                ff.write("\n".join(output_string))
            with open(f"DES5YR_{n}_ERRORFUDGES_{k1}_{k2}_PSF_{cut}.DAT", "w") as ff:
                ff.write("\n".join(output_string))
                ff.write("\n".join(output_string_pdf))
