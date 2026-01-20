import inspect
import shutil
import subprocess

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic

from pippin.classifiers.classifier import Classifier
from pippin.config import mkdirs, get_output_loc, ensure_list
from pippin.dataprep import DataPrep
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from pippin.task import Task
import pandas as pd
import os
from astropy.io import fits
import numpy as np


class Aggregator(Task):
    """Merge fitres files and aggregator output

    CONFIGURATION:
    ==============
    AGGREGATION:
      AGG:
        MASK: TEST  # partial match on sim and classifier
        MASK_SIM: TEST  # partial match on sim name
        MASK_CLAS: TEST  # partial match on classifier name
        RECALIBRATION: SIMNAME # To recalibrate on.
        OPTS:
          PLOT: True  # Whether or not to generate the PR curve, ROC curve, reliability plot, etc. Can specify a PYTHON FILE WHICH GETS INVOKED
          PLOT_ALL: False # If you use RANSEED_CHANGE, should we plot for all versions. Defaults to no.
          # PLOT: True will default to external/aggregator_plot.py, copy that for customisation

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        classifier_names: aggregators classifier names
        classifier_merge: Merge map of classifier task to prob column name
        merge_predictions_filename: location of the merged csv file
        merge_key_filename: location of the merged fitres file
        sn_column_name: name of the SNID column
        sn_type_name: name of type column, only exists if INCLUDE_TYPE was set
        sim_name: name of sim
        lcfit_names: names of the lcfit tasks being merged
        calibration_files: list[str] - all the calibration files. Hopefully only one will be made if you havent done something weird with the config
        empty_agg: if there were no types or ids that could be found.
    """

    def __new__(cls, name, output_dir, config, dependencies, options, recal_aggtask):
        # XXX DEPRECATION
        # If no BASE file is present, run legacy version of Aggregator
        # Avoid recursive nonsense by making sure the type of `cls` is Aggregator
        if cls == Aggregator and config.get("BASE") is None:
            # Have to import later because Aggregator must exist prior to importing AggregatorLegacy
            from pippin.aggregator_legacy import AggregatorLegacy

            cls = AggregatorLegacy
        return super().__new__(cls)

    def __init__(self, name, output_dir, config, dependencies, options, recal_aggtask):
        super().__init__(name, output_dir, config=config, dependencies=dependencies)
        self.passed = False
        self.classifiers = [d for d in dependencies if isinstance(d, Classifier)]
        self.lcfit_deps = [
            d for c in self.classifiers for d in c.get_fit_dependency(output=False)
        ]
        self.lcfit_names = list(
            set([l.output["name"] for l in self.lcfit_deps if l is not None])
        )
        self.output["lcfit_names"] = self.lcfit_names
        if not self.lcfit_names:
            self.logger.debug("No jobs depend on the LCFIT, so adding a dummy one")
            self.lcfit_names = [""]

        self.sim_task = self.get_underlying_sim_task()
        self.output["sim_name"] = self.sim_task.name
        self.recal_aggtask = recal_aggtask
        self.output["agg_name"] = self.name
        self.num_versions = len(self.sim_task.output["sim_folders"])

        self.output_dfs = [
            os.path.join(self.output_dir, f"merged_{i}.csv")
            for i in range(self.num_versions)
        ]
        self.output_dfs_key = [
            [
                os.path.join(self.output_dir, f"merged_{l}_{i}.key")
                for l in self.lcfit_names
            ]
            for i in range(self.num_versions)
        ]
        self.output_cals = [
            os.path.join(self.output_dir, f"calibration_{i}.csv")
            for i in range(self.num_versions)
        ]

        self.id = "CID"
        self.type_name = "SNTYPE"
        self.options = options
        self.include_type = bool(options.get("INCLUDE_TYPE", False))
        self.plot = options.get("PLOT", False)
        self.plot_all = options.get("PLOT_ALL", False)
        self.output["classifier_names"] = [c.name for c in self.classifiers]
        self.output["classifier_indexes"] = [c.index for c in self.classifiers]
        self.output["calibration_files"] = self.output_cals
        self.output["empty_agg"] = False
        if isinstance(self.plot, bool):
            self.python_file = (
                os.path.dirname(inspect.stack()[0][1]) + "/external/aggregator_plot.py"
            )
        else:
            self.python_file = self.plot
        self.python_file = get_output_loc(self.python_file)

        if not os.path.exists(self.python_file):
            Task.fail_config(
                f"Attempting to find python file {self.python_file} but it's not there!"
            )

        merge_classifiers = self.config.get("MERGE_CLASSIFIERS")
        self.classifier_merge = {
            c.output["name"]: c.get_prob_column_name() for c in self.classifiers
        }
        if merge_classifiers is not None:
            self.classifier_merge = dict()
            for c in self.classifiers:
                prob_col = []
                for prob_col_name in merge_classifiers.keys():
                    mask_list = ensure_list(merge_classifiers[prob_col_name])
                    match = False
                    for m in mask_list:
                        if match:
                            continue
                        else:
                            if m in c.output["name"]:
                                match = True
                    if match:
                        if prob_col_name[:5] != "PROB_":
                            prob_col_name = "PROB_" + prob_col_name
                        prob_col.append(prob_col_name)
                if len(prob_col) == 1:
                    self.classifier_merge[c.output["name"]] = prob_col[0]
                else:
                    if len(prob_col) == 0:
                        self.classifier_merge[c.output["name"]] = (
                            c.get_prob_column_name()
                        )
                    else:
                        Task.fail_config(
                            f"Classifier task {c.output['name']} matched multiple MERGE_CLASSIFIERS keys: {prob_col}. Please provide more specific keys"
                        )
        self.logger.debug(f"Classifier merge = {self.classifier_merge}")
        self.output["classifier_merge"] = self.classifier_merge

    def _check_completion(self, squeue):
        if not self.passed:
            self.logger.debug(
                "Task not reporting passed, might be external. Checking done file."
            )
            if os.path.exists(self.done_file):
                self.logger.debug("Done file exists, loading contents")
                with open(self.done_file) as f:
                    self.passed = "SUCCESS" in f.read()
                    self.logger.debug(
                        f"After reading done file, passed set to {self.passed}"
                    )
            else:
                self.logger.warning(
                    f"Task has not set passed and has no done file at {self.done_file}, returning failure"
                )
        return Task.FINISHED_SUCCESS if self.passed else Task.FINISHED_FAILURE

    def get_underlying_sim_task(self):
        check = []
        for task in self.dependencies:
            for t in task.dependencies:
                check.append(t)
                if isinstance(task, SNANALightCurveFit):
                    check += task.dependencies

        for task in check + self.dependencies:
            if isinstance(task, SNANASimulation) or isinstance(task, DataPrep):
                return task
        self.logger.error(
            f"Unable to find a simulation or data dependency for aggregator {self.name}"
        )
        return None

    def load_prediction_file(self, filename):
        df = pd.read_csv(filename, comment="#")
        columns = df.columns
        if len(columns) == 1 and "VARNAME" in columns[0]:
            df = pd.read_csv(filename, comment="#", delim_whitespace=True)
        if "VARNAMES:" in df.columns:
            df = df.drop(columns="VARNAMES:")
        remove_columns = [
            c for i, c in enumerate(df.columns) if i != 0 and "PROB_" not in c
        ]
        df = df.drop(columns=remove_columns)
        return df

    def save_calibration_curve(self, df, output_name):
        self.logger.debug("Creating calibration curves")

        # First let us define some prob bins
        bins = np.linspace(
            -1, 2, 61
        )  # Yes, outside normal range, so if we smooth it we dont screw things up with edge effects
        bc = 0.5 * (bins[:-1] + bins[1:])
        mask = (bc >= 0) & (bc <= 1)
        bc3 = bc[mask]  # Dont bother saving out the negative probs
        bc4 = np.concatenate(([0], bc3, [1.0]))  # For them bounds

        truth = df["IA"]
        truth_mask = np.isfinite(truth)
        cols = [c for c in df.columns if c.startswith("PROB_")]
        result = {"bins": bc4}
        for c in cols:
            data = df[c]

            if data.isnull().sum() == data.size or truth.isnull().sum() == truth.size:
                self.logger.warning(
                    "Unable to create calibration curves. This is expected if the calibration source is data (unknown types) or a sim of only Ia or CC (where you have only one type)."
                )
                if data.isnull().sum() == data.size:
                    self.logger.error(f"prob column {c} is all NaN")
                if truth.isnull().sum() == truth.size:
                    self.logger.error("truth values are all NaN")
                continue

            # Remove NaNs
            data_mask = np.isfinite(data)
            combined_mask = truth_mask & data_mask
            if combined_mask.sum() < 100:
                if combined_mask.sum() == 0:
                    self.logger.warning(
                        "There are no events which have both a prob and a known Ia/CC flag"
                    )
                else:
                    self.logger.warning(
                        "There are too few events with both a prob and known Ia/CC flag"
                    )
                continue

            data2 = data[combined_mask]
            truth2 = truth[combined_mask].astype(float)

            actual_prob, _, _ = binned_statistic(
                data2, truth2, bins=bins, statistic="mean"
            )
            m = np.isfinite(actual_prob)  # All the -1 to 0 and 1 to 2 probs will be NaN

            # Sets a 1:1 line outside of 0 to 1
            actual_prob2 = actual_prob.copy()
            actual_prob2[~m] = bc[~m]

            # Optional Gaussian filter. Turning this off but I'll keep the code in, in case we want to play around later with changing sigma
            actual_prob3 = gaussian_filter(actual_prob2, sigma=0)[mask]

            # Lets make sure out bounds are correct and all prob values are good. The last two lines will only do anything if sigma>0
            actual_prob4 = np.concatenate(([0], actual_prob3, [1.0]))
            actual_prob4[actual_prob4 < 0] = 0
            actual_prob4[actual_prob4 > 1] = 1
            result[c] = actual_prob4

        result_df = pd.DataFrame(result)
        result_df.to_csv(output_name, index=False)
        self.logger.debug(f"Calibration curves output to {output_name}")

    def recalibrate(self, df):
        self.logger.debug("Recalibrating!")
        curves = self.load_calibration_curve()
        cols = [c for c in df.columns if c.startswith("PROB_")]
        for c in cols:
            self.logger.debug(f"Recalibrating column {c}")
            data = df[c]
            if c not in curves:
                self.logger.warning(
                    f"Classifier {c} cannot be recalibrated. If this is because its FITPROB or another fake classifier, all good."
                )
                recalibrated = data
            else:
                recalibrated = interp1d(curves["bins"], curves[c])(data)
            df[c.replace("PROB_", "CPROB_")] = recalibrated
        self.logger.debug(
            "Returning recalibrated curves. They start with CPROB_, instead of PROB_"
        )
        return df

    def load_calibration_curve(self):
        path = self.recal_aggtask.output["calibration_files"]
        if len(path) > 1:
            self.logger.warning(
                f"Warning, found multiple calibration files, only using first one: {path}"
            )
        assert len(path) != 0, (
            f"No calibration files found for agg task {self.recal_aggtask}"
        )
        path = path[0]

        df = pd.read_csv(path)
        self.logger.debug(f"Reading calibration curves from {path}")
        return df

    def _run(
        self,
    ):
        new_hash = self.get_hash_from_string(
            self.name + str(self.include_type) + str(self.plot)
        )
        if self._check_regenerate(new_hash):
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)

            # Want to loop over each number and grab the relevant IDs and classifiers
            for index in range(self.num_versions):
                relevant_classifiers = [c for c in self.classifiers if c.index == index]
                self.logger.debug(f"relevant_classifiers: {relevant_classifiers}")

                prediction_files = [
                    d.output["predictions_filename"] for d in relevant_classifiers
                ]
                lcfits = [d.get_fit_dependency() for d in relevant_classifiers]
                self.logger.debug(f"lcfits: {lcfits}")

                df = None

                colnames = [self.classifier_merge[d.name] for d in relevant_classifiers]
                self.logger.debug(f"colnames: {colnames}")
                need_to_rename = len(colnames) != len(set(colnames))
                rename_ind = []
                if need_to_rename:
                    self.logger.info(
                        "Detected duplicate probability column names, will need to rename them"
                    )
                    for i, n in enumerate(colnames):
                        if (
                            len([j for j in range(len(colnames)) if colnames[j] == n])
                            > 1
                        ):
                            rename_ind.append(i)

                for i, (f, d, l) in enumerate(
                    zip(prediction_files, relevant_classifiers, lcfits)
                ):
                    self.logger.debug(f"l: {l}")
                    dataframe = self.load_prediction_file(f)
                    dataframe = dataframe.rename(
                        columns={
                            d.get_prob_column_name(): self.classifier_merge[d.name]
                        }
                    )
                    dataframe = dataframe.rename(
                        columns={dataframe.columns[0]: self.id}
                    )
                    dataframe[self.id] = dataframe[self.id].apply(str)
                    dataframe[self.id] = dataframe[self.id].str.strip()
                    if (
                        need_to_rename
                        and (l is not None or l != [])
                        and i in rename_ind
                    ):
                        lcname = ensure_list(l)[0]["name"]
                        self.logger.debug(
                            f"Renaming column {self.classifier_merge[d.name]} to include LCFIT name {lcname}"
                        )
                        dataframe = dataframe.rename(
                            columns={
                                self.classifier_merge[d.name]: self.classifier_merge[
                                    d.name
                                ]
                                + "_RENAMED_"
                                + lcname
                            }
                        )
                    self.logger.debug(f"Merging on column {self.id} for file {f}")
                    if df is None:
                        df = dataframe
                    else:
                        df = pd.merge(df, dataframe, on=self.id, how="outer")

                self.logger.info(
                    f"Finding original types, size of prediction df is {df.shape if df is not None else 'None'}"
                )
                s = self.get_underlying_sim_task()
                type_df = None
                phot_dir = s.output["photometry_dirs"][index]
                headers = [
                    os.path.join(phot_dir, a)
                    for a in os.listdir(phot_dir)
                    if "HEAD" in a
                ]
                if len(headers) == 0:
                    self.logger.warning(
                        f"No HEAD fits files found in {phot_dir}, manually running grep command!"
                    )

                    cmd = "grep --exclude-dir=* TYPE *.dat | awk -F ':' '{print $1 $3}'"
                    self.logger.debug(f"Running command   {cmd}  in dir {phot_dir}")
                    process = subprocess.run(
                        cmd, capture_output=True, cwd=phot_dir, shell=True
                    )
                    output = process.stdout.decode("ascii").split("\n")
                    output = [x for x in output if x]

                    cmd = "zgrep TYPE *.dat.gz | awk -F ':' '{print $1 $3}'"
                    self.logger.debug(f"Running command  {cmd}  in dir {phot_dir}")
                    process = subprocess.run(
                        cmd, capture_output=True, cwd=phot_dir, shell=True
                    )
                    output2 = process.stdout.decode("ascii").split("\n")
                    output += [x for x in output2 if x]

                    cmd = "zgrep TYPE *.txt | awk -F ':' '{print $1 $3}'"
                    self.logger.debug(f"Running command  {cmd}  in dir {phot_dir}")
                    process = subprocess.run(
                        cmd, capture_output=True, cwd=phot_dir, shell=True
                    )
                    output3 = process.stdout.decode("ascii").split("\n")
                    output += [x for x in output3 if x]

                    if len(output) == 0:
                        snid = []
                    else:
                        if "_" in output[0]:  # check if photometry is in filename
                            snid = [
                                x.split()[0].split("_")[1].split(".")[0] for x in output
                            ]
                            snid = [x[1:] if x.startswith("0") else x for x in snid]
                        else:
                            snid = [x.split()[0].split(".")[0] for x in output]
                            snid = [x[1:] if x.startswith("0") else x for x in snid]
                    sntype = [x.split()[1].strip() for x in output]

                    type_df = pd.DataFrame({self.id: snid, self.type_name: sntype})
                    type_df[self.id] = type_df[self.id].astype(str).str.strip()
                    type_df.drop_duplicates(subset=self.id, inplace=True)
                else:
                    for h in headers:
                        with fits.open(h) as hdul:
                            data = hdul[1].data
                            snid = np.array(data.field("SNID"))
                            sntype = np.array(data.field("SNTYPE")).astype(np.int64)
                            # self.logger.debug(f"Photometry has fields {hdul[1].columns.names}")
                            dataframe = pd.DataFrame(
                                {self.id: snid, self.type_name: sntype}
                            )
                            dataframe[self.id] = (
                                dataframe[self.id].astype(str).str.strip()
                            )
                            if type_df is None:
                                type_df = dataframe
                            else:
                                type_df = pd.concat([type_df, dataframe])
                        type_df.drop_duplicates(subset=self.id, inplace=True)
                    self.logger.debug(
                        f"Photometric types are {type_df['SNTYPE'].unique()}"
                    )

                if type_df is not None:
                    if df is None:
                        self.logger.debug("No original df found, only saving types")
                        df = type_df
                    else:
                        self.logger.debug(
                            f"Merging types of shape {type_df.shape} into df {df.shape}"
                        )
                        df = pd.merge(df, type_df, on=self.id, how="left")

                self.logger.debug(
                    f"Final dataframe from file ingestion has shape {df.shape}"
                )
                types = self.get_underlying_sim_task().output["types_dict"]
                has_nonia = len(types.get("NONIA", [])) > 0
                has_ia = len(types.get("IA", [])) > 0
                self.logger.debug(f"Input types are {types}")
                ia = df["SNTYPE"].apply(
                    lambda y: 1.0
                    if y in types["IA"]
                    else (0.0 if y in types["NONIA"] else np.nan)
                )
                df["IA"] = ia

                num_ia = (ia == 1.0).sum()
                num_cc = (ia == 0.0).sum()
                num_nan = ia.isnull().sum()

                self.logger.info(
                    f"Truth type has {num_ia} Ias, {num_cc} CCs and {num_nan} unknowns"
                )

                sorted_columns = [self.id, "SNTYPE", "IA"] + sorted(
                    [c for c in df.columns if c.startswith("PROB_")]
                )
                df = df.reindex(sorted_columns, axis=1)
                self.logger.info(
                    f"Merged into dataframe of {df.shape[0]} rows, with columns {list(df.columns)}"
                )

                if df.shape[0] == 0:
                    self.logger.warning(
                        "Oh no, dataframe doesnt have any rows. What is going on? What strange data format is this?"
                    )
                    self.output["empty_agg"] = True

                if has_nonia and has_ia:
                    self.save_calibration_curve(df, self.output_cals[index])
                    if self.recal_aggtask:
                        df = self.recalibrate(df)

                df.to_csv(self.output_dfs[index], index=False, float_format="%0.4f")

                for l in self.lcfit_names:
                    self.save_key_format(df, index, l)
                self.logger.debug(
                    f"Saving merged dataframe to {self.output_dfs[index]}"
                )
                self.save_new_hash(new_hash)

                if self.plot:
                    if index == 0 or self.plot_all:
                        return_good = self._plot(index)
                        if not return_good:
                            self.logger.error(
                                "Plotting did not work correctly! Attempting to continue anyway."
                            )
                else:
                    self.logger.debug("Plot not set, skipping plotting section")

            # Write the done file
            self.logger.debug(f"Writing done file to {self.done_file}")
            with open(self.done_file, "w") as f:
                f.write("SUCCESS")

        else:
            self.should_be_done()
            self.logger.info("Hash check passed, not rerunning")

        self.output["merge_predictions_filename"] = self.output_dfs
        self.output["merge_key_filename"] = self.output_dfs_key
        self.output["sn_column_name"] = self.id
        if self.include_type:
            self.output["sn_type_name"] = self.type_name

        self.passed = True

        return True

    def save_key_format(self, df, index, lcfitname):
        lc_index = (
            0 if len(self.lcfit_names) == 1 else self.lcfit_names.index(lcfitname)
        )
        if "IA" in df.columns:
            df = df.drop(columns=[self.type_name, "IA"])
        cols_to_rename = [c for c in df.columns if "_RENAMED_" in c]
        for c in cols_to_rename:
            name, lcfit = c.split("_RENAMED_")
            if lcfit == lcfitname:
                df = df.rename(columns={c: name})
            else:
                self.logger.warning(
                    f"Aggregation {self.name} for LCFIT {lcfitname} is dropping column {c} as it doesnt match."
                )
                df = df.drop(columns=[c])
        df2 = df.fillna(0.0)
        df2.insert(0, "VARNAMES:", ["SN:"] * df2.shape[0])
        df2.to_csv(
            self.output_dfs_key[index][lc_index],
            index=False,
            float_format="%0.4f",
            sep=" ",
        )

    def _plot(self, index):
        cmd = [
            "python",
            self.python_file,
            self.output_dfs[index],
            self.output_dir,
            f"{index}",
        ]
        self.logger.debug(f"Invoking command  {' '.join(cmd)}")
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.output_dir,
                check=True,
            )
            self.logger.info(f"Finished invoking {self.python_file}")
        except subprocess.CalledProcessError:
            return False
        return True

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        sim_tasks = Task.get_task_of_type(prior_tasks, SNANASimulation, DataPrep)
        classifier_tasks = Task.get_task_of_type(prior_tasks, Classifier)

        def _get_aggregator_dir(base_output_dir, stage_number, agg_name):
            return f"{base_output_dir}/{stage_number}_AGG/{agg_name}"

        tasks = []

        # Check for recalibration, and if so, find that task first
        for agg_name in c.get("AGGREGATION", []):
            config = c["AGGREGATION"][agg_name]
            if config is None:
                config = {}
            options = config.get("OPTS", {})
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_clas = config.get("MASK_CLAS", "")
            recalibration = config.get("RECALIBRATION")
            recal_simtask = None
            recal_aggtask = None
            if recalibration:
                recal_sim = [
                    i for i, s in enumerate(sim_tasks) if s.name == recalibration
                ]

                if len(recal_sim) == 0:
                    Task.fail_config(
                        f"Recalibration sim {recalibration} not in the list of available sims: {[s.name for s in sim_tasks]}"
                    )
                elif len(recal_sim) > 1:
                    Task.fail_config(
                        f"Recalibration aggregation {recalibration} not in the list of available aggs: {[s.name for s in sim_tasks]}"
                    )

                # Move the recal sim task to the front of the queue so it executes first
                recal_sim_index = recal_sim[0]
                recal_simtask = sim_tasks[recal_sim_index]
                sim_tasks.insert(0, sim_tasks.pop(recal_sim_index))

            for sim_task in sim_tasks:
                if (
                    mask_sim not in sim_task.name
                    or mask not in sim_task.name
                    and recal_simtask != sim_task
                ):
                    continue

                agg_name2 = f"{agg_name}_{sim_task.name}"
                deps = [
                    c
                    for c in classifier_tasks
                    if mask in c.name
                    and mask_clas in c.name
                    and c.mode == Classifier.PREDICT
                    and sim_task in c.get_simulation_dependency()
                ]
                if len(deps) == 0:
                    deps = [sim_task]

                if recalibration and sim_task != recal_simtask:
                    if recal_aggtask is None:
                        Task.fail_config(
                            f"The aggregator task for {recalibration} has not been made yet. Sam probably screwed up dependency order."
                        )
                    else:
                        deps.append(recal_aggtask)
                a = Aggregator(
                    agg_name2,
                    _get_aggregator_dir(base_output_dir, stage_number, agg_name2),
                    config,
                    deps,
                    options,
                    recal_aggtask,
                )
                if sim_task == recal_simtask:
                    recal_aggtask = a
                Task.logger.info(
                    f"Creating aggregation task {agg_name2} for {sim_task.name} with {a.num_jobs} jobs"
                )
                tasks.append(a)

        return tasks
