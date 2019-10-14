import inspect
import shutil
import subprocess

from pippin.classifiers.classifier import Classifier
from pippin.config import mkdirs, get_output_loc
from pippin.dataprep import DataPrep
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from pippin.task import Task
import pandas as pd
import os
from astropy.io import fits
import numpy as np


class Aggregator(Task):
    """ Merge fitres files and aggregator output

    CONFIGURATION:
    ==============
    AGGREGATION:
      AGG:
        MASK: TEST  # partial match on sim and classifier
        MASK_SIM: TEST  # partial match on sim name
        MASK_CLAS: TEST  # partial match on classifier name
        OPTS:
          PLOT: True  # Whether or not to generate the PR curve, ROC curve, reliability plot, etc. Can specify a PYTHON FILE WHICH GETS INVOKED
          PLOT_ALL: False # If you use RANSEED_CHANGE, should we plot for all versions. Defaults to no.
          # PLOT: True will default to external/aggregator_plot.py, copy that for customisation

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        classifiers: classifier tasks
        classifier_names: aggregators classifier names
        merge_predictions_filename: location of the merged csv file
        merge_key_filename: location of the merged fitres file
        sn_column_name: name of the SNID column
        sn_type_name: name of type column, only exists if INCLUDE_TYPE was set

    """

    def __init__(self, name, output_dir, dependencies, options):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.passed = False
        self.classifiers = [d for d in dependencies if isinstance(d, Classifier)]

        self.sim_task = self.get_underlying_sim_task()
        self.num_versions = len(self.sim_task.output["sim_folders"])

        self.output_dfs = [os.path.join(self.output_dir, f"merged_{i}.csv") for i in range(self.num_versions)]
        self.output_dfs_key = [os.path.join(self.output_dir, f"merged_{i}.key") for i in range(self.num_versions)]

        self.id = "CID"
        self.type_name = "SNTYPE"
        self.options = options
        self.include_type = bool(options.get("INCLUDE_TYPE", False))
        self.plot = options.get("PLOT", True)
        self.plot_all = options.get("PLOT_ALL", False)
        self.output["classifiers"] = self.classifiers

        if isinstance(self.plot, bool):
            self.python_file = os.path.dirname(inspect.stack()[0][1]) + "/external/aggregator_plot.py"
        else:
            self.python_file = self.plot
        self.python_file = get_output_loc(self.python_file)

        if not os.path.exists(self.python_file):
            Task.fail_config(f"Attempting to find python file {self.python_file} but it's not there!")

    def _check_completion(self, squeue):
        return Task.FINISHED_SUCCESS if self.passed else Task.FINISHED_FAILURE

    def check_regenerate(self, force_refresh):
        new_hash = self.get_hash_from_string(self.name + str(self.include_type) + str(self.plot))
        old_hash = self.get_old_hash(quiet=True)

        if new_hash != old_hash:
            self.logger.info("Hash check failed, regenerating")
            return new_hash
        elif force_refresh:
            self.logger.debug("Force refresh deteted")
            return new_hash
        else:
            self.logger.info("Hash check passed, not rerunning")
            return False

    def get_underlying_sim_task(self):
        check = []
        for task in self.dependencies:
            for t in task.dependencies:
                check.append(t)
                if isinstance(task, SNANALightCurveFit):
                    check += task.dependencies

        for task in check:
            if isinstance(task, SNANASimulation) or isinstance(task, DataPrep):
                return task
        self.logger.error(f"Unable to find a simulation or data dependency for aggregator {self.name}")
        return None

    def load_prediction_file(self, filename):
        df = pd.read_csv(filename, comment="#")
        columns = df.columns
        if len(columns) == 1 and "VARNAME" in columns[0]:
            df = pd.read_csv(filename, comment="#", sep=r"\s+")
        if "VARNAMES:" in df.columns:
            df = df.drop(columns="VARNAMES:")
        remove_columns = [c for i, c in enumerate(df.columns) if i != 0 and "PROB_" not in c]
        df = df.drop(columns=remove_columns)
        return df

    def _run(self, force_refresh):
        new_hash = self.check_regenerate(force_refresh)
        if new_hash:
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)

            # Want to loop over each number and grab the relevant IDs and classifiers
            for index in range(self.num_versions):
                relevant_classifiers = [c for c in self.classifiers if c.index == index]

                prediction_files = [d.output["predictions_filename"] for d in relevant_classifiers]
                df = None

                for f in prediction_files:
                    dataframe = self.load_prediction_file(f)
                    dataframe = dataframe.rename(columns={dataframe.columns[0]: self.id})
                    dataframe[self.id] = dataframe[self.id].apply(str)
                    dataframe[self.id] = dataframe[self.id].str.strip()
                    self.logger.debug(f"Merging on column {self.id} for file {f}")
                    if df is None:
                        df = dataframe
                    else:
                        df = pd.merge(df, dataframe, on=self.id, how="outer")  # Inner join atm, should I make this outer?

                self.logger.info("Finding original types")
                s = self.get_underlying_sim_task()
                type_df = None
                phot_dir = s.output["photometry_dirs"][index]
                headers = [os.path.join(phot_dir, a) for a in os.listdir(phot_dir) if "HEAD" in a]
                if not headers:
                    self.logger.error(f"No HEAD fits files found in {phot_dir}!")
                else:
                    for h in headers:
                        with fits.open(h) as hdul:
                            data = hdul[1].data
                            snid = np.array(data.field("SNID"))
                            sntype = np.array(data.field("SNTYPE")).astype(np.int64)
                            # self.logger.debug(f"Photometry has fields {hdul[1].columns.names}")
                            dataframe = pd.DataFrame({self.id: snid, self.type_name: sntype})
                            dataframe[self.id] = dataframe[self.id].apply(str)
                            dataframe[self.id] = dataframe[self.id].str.strip()
                            if type_df is None:
                                type_df = dataframe
                            else:
                                type_df = pd.concat([type_df, dataframe])
                        type_df.drop_duplicates(subset=self.id, inplace=True)
                    self.logger.debug(f"Photometric types are {type_df['SNTYPE'].unique()}")

                df = pd.merge(df, type_df, on=self.id, how="left")

                types = self.get_underlying_sim_task().output["types_dict"]
                self.logger.debug(f"Input types are {types}")
                ia = df["SNTYPE"].apply(lambda y: True if y in types["IA"] else (False if y in types["NONIA"] else np.nan))
                df["IA"] = ia

                sorted_columns = [self.id, "SNTYPE", "IA"] + sorted([c for c in df.columns if c.startswith("PROB_")])
                df = df.reindex(sorted_columns, axis=1)
                self.logger.info(f"Merged into dataframe of {df.shape[0]} rows, with columns {list(df.columns)}")
                df.to_csv(self.output_dfs[index], index=False, float_format="%0.4f")
                self.save_key_format(df, index)
                self.logger.debug(f"Saving merged dataframe to {self.output_dfs[index]}")
                self.save_new_hash(new_hash)

                if self.plot:
                    if index == 0 or self.plot_all:
                        return_good = self._plot(index)
                        if not return_good:
                            self.logger.error("Plotting did not work correctly! Attempting to continue anyway.")
                else:
                    self.logger.debug("Plot not set, skipping plotting section")

        self.output["merge_predictions_filename"] = self.output_dfs
        self.output["merge_key_filename"] = self.output_dfs_key
        self.output["sn_column_name"] = self.id
        if self.include_type:
            self.output["sn_type_name"] = self.type_name

        self.passed = True
        return True

    def save_key_format(self, df, index):
        if "IA" in df.columns:
            df = df.drop(columns=[self.type_name, "IA"])
        df2 = df.fillna(0.0)
        df2.insert(0, "VARNAMES:", ["SN:"] * df2.shape[0])
        df2.to_csv(self.output_dfs_key[index], index=False, float_format="%0.4f", sep=" ")

    def _plot(self, index):
        cmd = ["python", self.python_file, self.output_dfs[index], self.output_dir, f"{index}"]
        self.logger.debug(f"Invoking command  {' '.join(cmd)}")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.output_dir, check=True)
            self.logger.info(f"Finished invoking {self.python_file}")
        except subprocess.CalledProcessError as e:
            return False
        return True

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        sim_tasks = Task.get_task_of_type(prior_tasks, SNANASimulation, DataPrep)
        classifier_tasks = Task.get_task_of_type(prior_tasks, Classifier)

        def _get_aggregator_dir(base_output_dir, stage_number, agg_name):
            return f"{base_output_dir}/{stage_number}_AGG/{agg_name}"

        def get_num_ranseed(sim_task, lcfit_task):
            if sim_task is not None:
                return len(sim_task.output["sim_folders"])
            if lcfit_task is not None:
                return len(sim_task.output["fitres_dirs"])
            raise ValueError("Classifier dependency has no sim_task or lcfit_task?")

        tasks = []

        for agg_name in c.get("AGGREGATION", []):
            config = c["AGGREGATION"][agg_name]
            if config is None:
                config = {}
            options = config.get("OPTS", {})
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_clas = config.get("MASK_CLAS", "")
            for sim_task in sim_tasks:
                if mask_sim not in sim_task.name or mask not in sim_task.name:
                    continue

                agg_name2 = f"{agg_name}_{sim_task.name}"
                deps = [
                    c
                    for c in classifier_tasks
                    if mask in c.name and mask_clas in c.name and c.mode == Classifier.PREDICT and c.get_simulation_dependency() == sim_task
                ]
                if len(deps) == 0:
                    Task.fail_config(f"Aggregator {agg_name2} with mask {mask} matched no classifier tasks for sim {sim_task}")
                else:
                    a = Aggregator(agg_name2, _get_aggregator_dir(base_output_dir, stage_number, agg_name2), deps, options)
                    Task.logger.info(f"Creating aggregation task {agg_name2} for {sim_task.name} with {a.num_jobs} jobs")
                    tasks.append(a)

        return tasks
