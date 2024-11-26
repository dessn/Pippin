import os
import subprocess
import json
import shutil
import pickle
from collections import OrderedDict
from pippin.classifiers.classifier import Classifier
from pippin.config import (
    chown_dir,
    mkdirs,
    get_config,
    get_output_loc,
    get_data_loc,
    merge_dict,
)
from pippin.task import Task
from time import sleep
import numpy as np


class SuperNNovaClassifier(Classifier):
    """Classification task for the SuperNNova classifier.

    CONFIGURATION
    =============

    CLASSIFICATION:
        label:
            MASK_SIM: mask  # partial match
            MASK_FIT: mask  # partial match
            MASK: mask  # partial match
            MODE: train/predict
            GPU: True # default
            CLEAN: True # Whether to remove processed directory, defaults to true
            OPTS:  # Options
                VARIANT: vanilla  #  a variant to train. "vanilla", "variational", "bayesian". Defaults to "vanilla"
                MODEL: someName # exact name of training classification task
                REDSHIFT: True # Use spec redshift, defaults to True
                NORM: global # Global is default, can also pick cosmo or perfilter or cosmo_quantile

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        prob_column_name: name of the column to get probabilities out of
        predictions_filename: location of csv filename with id/probs


    """

    def __init__(
        self,
        name,
        output_dir,
        config,
        dependencies,
        mode,
        options,
        index=0,
        model_name=None,
    ):
        super().__init__(
            name,
            output_dir,
            config,
            dependencies,
            mode,
            options,
            index=index,
            model_name=model_name,
        )
        self.global_config = get_config()
        self.dump_dir = output_dir + "/dump"
        self.job_base_name = os.path.basename(output_dir)
        self.gpu = config.get("GPU", True)
        self.tmp_output = None
        self.done_file = os.path.join(self.output_dir, "done_task.txt")
        self.done_file2 = os.path.join(self.output_dir, "done_task2.txt")
        self.variant = options.get("VARIANT", "vanilla").lower()
        # Redshift can be True, False, 'zpho', 'zspe', or 'none'
        redshift = options.get("REDSHIFT", "zspe")
        # Not sure how python deals with strings and bools, so just being careful
        if redshift == True:
            redshift = "zspe"
        elif redshift == False:
            redshift = "none"
        if redshift not in ["zpho", "zspe", "none"]:
            self.logger.warning(f"Unknown redshift option ['zpho', 'zspe', 'none']")
        self.redshift = redshift
        self.norm = options.get("NORM", "cosmo")
        self.cyclic = options.get("CYCLIC", True)
        self.seed = options.get("SEED", 0)
        self.clean = config.get("CLEAN", True)
        self.batch_size = options.get("BATCH_SIZE", 128)
        self.num_layers = options.get("NUM_LAYERS", 2)
        self.hidden_dim = options.get("HIDDEN_DIM", 32)
        # Must be a list
        self.list_filters = options.get("LIST_FILTERS", None)
        if self.list_filters is None:
            self.list_filters = ["DES-g", "DES-i", "DES-r", "DES-z"]
        assert isinstance(
            self.list_filters, list
        ), f"LIST_FILTERS must be a list, instead got {type(self.list_filters)}"
        # Can either be a yml dictionary or a str filepath to a txt files containing all mappings
        self.sntypes = options.get("SNTYPES", None)
        if self.sntypes is None:
            self.sntypes = {}
        elif isinstance(self.sntypes, str):
            sntypes_path = get_data_loc(self.sntypes)
            assert (
                sntypes_path is not None
            ), f"SNTYPES: {self.sntypes} does not resolve to a path."
            self.logger.debug(f"Reading in SNTYPES from {sntypes_path}")
            sntypes_raw = np.loadtxt(sntypes_path, dtype=str)
            self.sntypes = {i[0]: i[1] for i in sntypes_raw}
        assert isinstance(
            self.sntypes, dict
        ), f"SNTYPES must be a dict, instead got {type(self.sntypes)}"

        # Setup yml files
        self.data_yml_file = options.get("DATA_YML", None)
        self.output_data_yml = os.path.join(self.output_dir, "data.yml")
        self.classification_yml_file = options.get("CLASSIFICATION_YML", None)
        self.output_classification_yml = os.path.join(
            self.output_dir, "classification.yml"
        )
        # XOR - only runs if either but not both yml's are None
        if (self.data_yml_file is None) ^ (self.classification_yml_file is None):
            self.logger.error(
                f"If using yml inputs, both 'DATA_YML' (currently {self.data_yml} and 'CLASSIFICATION_YML' (currently {self.classification_yml}) must be provided"
            )
        elif self.data_yml_file is not None:
            with open(self.data_yml_file, "r") as f:
                self.data_yml = f.read()
            with open(self.classification_yml_file, "r") as f:
                self.classification_yml = f.read()
            self.has_yml = True
            self.variant = self.get_variant_from_yml(self.classification_yml)
        else:
            self.data_yml = None
            self.classification_yml = None
            self.has_yml = False

        self.batch_file = self.options.get("BATCH_FILE")
        if self.batch_file is not None:
            self.batch_file = get_data_loc(self.batch_file)
        self.batch_replace = self.options.get(
            "BATCH_REPLACE", self.global_config.get("BATCH_REPLACE", {})
        )

        self.validate_model()

        assert self.norm in [
            "global",
            "cosmo",
            "perfilter",
            "cosmo_quantile",
            "none",
        ], f"Norm option is set to {self.norm}, needs to be one of 'global', 'cosmo', 'perfilter', 'cosmo_quantile"
        assert self.variant in [
            "vanilla",
            "variational",
            "bayesian",
        ], f"Variant {self.variant} is not vanilla, variational or bayesian"
        self.slurm = """{sbatch_header}
        {task_setup}

        """
        self.conda_env = self.global_config["SuperNNova"]["conda_env"]
        self.path_to_classifier = get_output_loc(
            self.global_config["SuperNNova"]["location"]
        )

    def get_variant_from_yml(self, yml_file):
        if "model" in yml_file:
            self.logger.debug("Detected model in yml file")
            stripped = "".join(yml_file.split(" "))
            if "model:bayesian" in stripped:
                self.logger.debug("Detected bayesian model")
                return "bayesian"
            if "model:variational" in stripped:
                self.logger.debug("Detected variational model")
                return "variational"
        self.logger.debug("Defaulting variant to vanilla")
        return "vanilla"

    def update_yml(self):
        replace_dict = {
            "DONE_FILE": self.done_file,
            "DUMP_DIR": self.dump_dir,
            "RAW_DIR": self.raw_dir,
        }
        for key, value in replace_dict.items():
            self.data_yml = self.data_yml.replace(key, value)
            self.classification_yml = self.classification_yml.replace(key, value)

    def get_model_and_pred(self):
        max_tries = 100
        while max_tries > 0:
            self.logger.debug(f"Max Tries: {max_tries}")
            try:
                model_folder = self.dump_dir + "/models"
                files = [
                    f
                    for f in os.listdir(model_folder)
                    if os.path.isdir(os.path.join(model_folder, f))
                ]
                assert (
                    len(files) == 1
                ), f"Did not find singular output file: {str(files)}"
                saved_dir = os.path.abspath(os.path.join(model_folder, files[0]))

                subfiles = list(os.listdir(saved_dir))
                model_files = [f for f in subfiles if f.endswith(".pt")]
                if model_files:
                    model_file = os.path.join(saved_dir, model_files[0])
                    self.logger.debug(f"Found model file {model_file}")
                else:
                    self.logger.debug(
                        "No model found. Not an issue if you've specified a model."
                    )
                    model_file = None
                ending = (
                    "_aggregated.pickle"
                    if self.variant in ["variational", "bayesian"]
                    else ".pickle"
                )
                pred_files = [
                    f for f in subfiles if f.startswith("PRED") and f.endswith(ending)
                ]
                self.logger.debug(pred_files)
                pred_file = pred_files[0]
                self.logger.debug(f"Success after {100-max_tries} tries.")
                break
            except Exception as e:
                self.logger.debug(e)
                sleep(5)
                max_tries -= 1
        return model_file, os.path.join(saved_dir, pred_file)

    def train(self):
        return self.classify(True)

    def predict(self):
        return self.classify(False)

    def get_types(self):
        types = {}
        for t in self.get_simulation_dependency():
            for k, v in t.output["types"].items():
                if k not in types:
                    types[k] = v
        return types

    def classify(self, training):
        model = self.options.get("MODEL")
        model_path = ""
        if not training:
            assert (
                model is not None
            ), "If TRAIN is not specified, you have to point to a model to use"
            if not os.path.exists(get_output_loc(model)):
                for t in self.dependencies:
                    if model == t.name:
                        self.logger.debug(
                            f"Found task dependency {t.name} with model file {t.output['model_filename']}"
                        )
                        model = t.output["model_filename"]
            model_path = get_output_loc(model)
            self.logger.debug(f"Looking for model in {model_path}")
            assert os.path.exists(model_path), f"Cannot find {model_path}"

        # If you specify sntypes in the pippin input use that
        if (self.sntypes is not None) and (len(self.sntypes) > 0):
            types = self.sntypes
        else:
            # See if sntypes was defined by the SIM use that
            # Otherwise use default
            types = self.get_types()
            if types is None:
                types = OrderedDict(
                    {
                        1: "Ia",
                        0: "unknown",
                        2: "SNIax",
                        3: "SNIa-pec",
                        20: "SNIIP",
                        21: "SNIIL",
                        22: "SNIIn",
                        29: "SNII",
                        32: "SNIb",
                        33: "SNIc",
                        39: "SNIbc",
                        41: "SLSN-I",
                        42: "SLSN-II",
                        43: "SLSN-R",
                        80: "AGN",
                        81: "galaxy",
                        98: "None",
                        99: "pending",
                        101: "Ia",
                        120: "SNII",
                        130: "SNIbc",
                    }
                )
            else:
                has_ia = False
                has_cc = False
                self.logger.debug(f"Input types set to {types}")
                for key, value in types.items():
                    if value.upper() == "IA":
                        has_ia = True
                    elif value.upper() in ["II", "IBC"]:
                        has_cc = True
                if not has_ia:
                    self.logger.debug("No Ia type found, injecting type")
                    types[1] = "Ia"
                    types = dict(
                        sorted(types.items(), key=lambda x: -1 if x[0] == 1 else x[0])
                    )
                    self.logger.debug(f"Inject types with Ias are {types}")
                if not has_cc:
                    self.logger.debug("No cc type found, injecting type")
                    types[29] = "II"
        # Ensure appropriate Ia types are always included
        types[1] = "Ia"
        types[10] = "Ia"
        types[110] = "Ia"
        str_types = json.dumps(types)
        self.logger.debug(f"Types set to {str_types}")

        str_list_filters = " ".join(self.list_filters)
        self.logger.debug(f"Filter list set to {str_list_filters}")

        sim_dep = self.get_simulation_dependency()[
            0
        ]  # only taking the first one because SNN internally takes a single fits dir as input
        if len(self.get_simulation_dependency()) > 1:
            self.logger.warning(
                f"Found more than one simulation dependency, possibly because COMBINE_MASK is being used. SuperNNova doesn't currently support this.  Using only the first sim dependency: {sim_dep.name}"
            )
        light_curve_dir = sim_dep.output["photometry_dirs"][self.index]
        self.raw_dir = light_curve_dir
        fit = self.get_fit_dependency()
        fit_dir = (
            f""
            if ((fit is None) or (len(fit) == 0))
            else f"--fits_dir {fit[self.index]['fitres_dirs']}"
        )
        cyclic = (
            "--cyclic"
            if self.variant in ["vanilla", "variational"] and self.cyclic
            else ""
        )
        batch_size = f"--batch_size {self.batch_size}"
        num_layers = f"--num_layers {self.num_layers}"
        hidden_dim = f"--hidden_dim {self.hidden_dim}"
        variant = f"--model {self.variant}"
        if self.variant == "bayesian":
            variant += " --num_inference_samples 20"

        clump = sim_dep.output.get("clump_file")
        if clump is None:
            clump_txt = ""
        else:
            clump_txt = f"--photo_window_files {clump}"

        if self.batch_file is None:
            if self.gpu:
                self.sbatch_header = self.sbatch_gpu_header
            else:
                self.sbatch_header = self.sbatch_cpu_header
        else:
            with open(self.batch_file, "r") as f:
                self.sbatch_header = f.read()
            self.sbatch_header = self.clean_header(self.sbatch_header)

        if self.has_yml:
            self.update_yml()
            setup_file = "supernnova_yml"
        else:
            setup_file = "supernnova"

        header_dict = {
            "REPLACE_NAME": self.job_base_name,
            "REPLACE_WALLTIME": "23:00:00",
            "REPLACE_LOGFILE": "output.log",
            "REPLACE_MEM": "32GB",
            "APPEND": ["#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1"],
        }
        header_dict = merge_dict(header_dict, self.batch_replace)
        self.update_header(header_dict)

        setup_dict = {
            "conda_env": self.conda_env,
            "dump_dir": self.dump_dir,
            "photometry_dir": light_curve_dir,
            "fit_dir": fit_dir,
            "path_to_classifier": self.path_to_classifier,
            "job_name": self.job_base_name,
            "command": "--train_rnn" if training else "--validate_rnn",
            "sntypes": str_types,
            "list_filters": str_list_filters,
            "variant": variant,
            "cyclic": cyclic,
            "model": "" if training else f"--model_files {model_path}",
            "phot": "",
            "test_or_train": "" if training else "--data_testing",
            "redshift": "--redshift " + self.redshift,
            "norm": "--norm " + self.norm,
            "done_file": self.done_file,
            "clump": clump_txt,
            "done_file2": self.done_file2,
            "partition": "gpu2" if self.gpu else "broadwl",
            "gres": "#SBATCH --gres=gpu:1" if self.gpu else "",
            "cuda": "--use_cuda" if self.gpu else "",
            "clean_command": f"rm -rf {self.dump_dir}/processed" if self.clean else "",
            "seed": f"--seed {self.seed}" if self.seed else "",
            "batch_size": batch_size,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "data_yml": self.output_data_yml,
            "classification_yml": self.output_classification_yml,
            "classification_command": "train_rnn" if training else "validate_rnn",
        }

        format_dict = {
            "sbatch_header": self.sbatch_header,
            "task_setup": self.update_setup(setup_dict, self.task_setup[setup_file]),
        }

        slurm_output_file = self.output_dir + "/job.slurm"
        self.logger.info(
            f"Running SuperNNova, slurm job outputting to {slurm_output_file}"
        )
        slurm_text = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string(slurm_text)

        if not self._check_regenerate(new_hash):
            self.should_be_done()
        else:
            self.logger.info("Rerunning. Cleaning output_dir")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            if self.has_yml:
                with open(self.output_data_yml, "w") as f:
                    f.write(self.data_yml)
                with open(self.output_classification_yml, "w") as f:
                    f.write(self.classification_yml)

            self.save_new_hash(new_hash)

            with open(slurm_output_file, "w") as f:
                f.write(slurm_text)

            self.logger.info(
                f"Submitting batch job to {'train' if training else 'predict using'} SuperNNova"
            )
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)

        return True

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file) or os.path.exists(self.done_file2):
            self.logger.info("Job complete")
            if os.path.exists(self.done_file):
                with open(self.done_file) as f:
                    if "FAILURE" in f.read():
                        return Task.FINISHED_FAILURE
            if os.path.exists(self.done_file2):
                with open(self.done_file2) as f:
                    if "FAILURE" in f.read():
                        return Task.FINISHED_FAILURE

            new_pred_file = self.output_dir + "/predictions.csv"
            new_model_file = os.path.join(self.output_dir, f"model.pt")

            if not os.path.exists(new_pred_file) or not os.path.exists(new_model_file):
                self.logger.info(
                    "Updating model location or generating predictions file"
                )
                model, predictions = self.get_model_and_pred()

                if not os.path.exists(new_model_file):
                    if model is not None:
                        shutil.move(model, new_model_file)
                        args_old, args_new = (
                            os.path.abspath(
                                os.path.join(os.path.dirname(model), "cli_args.json")
                            ),
                            self.output_dir + "/cli_args.json",
                        )
                        norm_old, norm_new = (
                            os.path.abspath(
                                os.path.join(os.path.dirname(model), "data_norm.json")
                            ),
                            self.output_dir + "/data_norm.json",
                        )
                        shutil.move(args_old, args_new)
                        shutil.move(norm_old, norm_new)
                        self.logger.info(f"Model file can be found at {new_model_file}")
                if not os.path.exists(new_pred_file):
                    with open(predictions, "rb") as f:
                        dataframe = pickle.load(f)
                        self.logger.debug(dataframe)
                        self.logger.debug(self.variant)
                        if self.variant in ["variational", "bayesian"]:
                            final_dataframe = dataframe[
                                ["SNID", "all_class0_median", "all_class0_std"]
                            ]
                            final_dataframe = final_dataframe.rename(
                                columns={
                                    "all_class0_median": self.get_prob_column_name(),
                                    "all_class0_std": self.get_prob_column_name()
                                    + "_ERR",
                                }
                            )
                        else:
                            final_dataframe = dataframe[["SNID", "all_class0"]]
                            final_dataframe = final_dataframe.rename(
                                columns={"all_class0": self.get_prob_column_name()}
                            )
                        final_dataframe.to_csv(
                            new_pred_file, index=False, float_format="%0.4f"
                        )
                        self.logger.info(
                            f"Predictions file can be found at {new_pred_file}"
                        )
                chown_dir(self.output_dir)

            self.output.update(
                {
                    "model_filename": new_model_file,
                    "predictions_filename": new_pred_file,
                }
            )
            return Task.FINISHED_SUCCESS
        else:
            return self.check_for_job(squeue, self.job_base_name)

    @staticmethod
    def get_requirements(options):
        return True, False
