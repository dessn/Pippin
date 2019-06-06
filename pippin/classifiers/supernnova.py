import os
import subprocess
import json
import shutil
import pickle
from collections import OrderedDict

from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs, get_config, get_output_loc
from pippin.task import Task


class SuperNNovaClassifier(Classifier):
    """ Classification task for the SuperNNova classifier.

    Current valid options are specific to SuperNNova are:

        USE_PHOTOMETRY - Use only photometry and no fitres summaries
        VARIANT - a variant to train. "vanilla", "variational", "bayesian". Defaults to "vanilla"

    Global classification options:

        MODEL - a task name or file location with a trained model to use when predicting.
    """

    def __init__(self, name, output_dir, dependencies, mode, options):
        super().__init__(name, output_dir, dependencies, mode, options)
        self.global_config = get_config()
        self.dump_dir = output_dir + "/dump"
        self.job_base_name = os.path.basename(output_dir)

        self.tmp_output = None
        self.done_file = os.path.join(self.output_dir, "done_task.txt")
        self.variant = options.get("VARIANT", "vanilla").lower()
        assert self.variant in ["vanilla", "variational", "bayesian"], \
            f"Variant {self.variant} is not vanilla, variational or bayesian"
        self.slurm = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=output.log
#SBATCH --account=pi-rkessler
#SBATCH --mem=64GB

source activate {conda_env}
module load cuda
echo `which python`
cd {path_to_classifier}
python run.py --data --sntypes '{sntypes}' --dump_dir {dump_dir} --raw_dir {photometry_dir} {fit_dir} {phot} {clump} {test_or_train}
python run.py --use_cuda {cyclic} --sntypes '{sntypes}' --done_file {done_file} --dump_dir {dump_dir} {cyclic} {variant} {model} {phot} {command}
        """
        self.conda_env = self.global_config["SuperNNova"]["conda_env"]
        self.path_to_classifier = get_output_loc(self.global_config["SuperNNova"]["location"])

    def get_model_and_pred(self):
        model_folder = self.dump_dir + "/models"
        files = [f for f in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, f))]
        assert len(files) == 1, f"Did not find singular output file: {str(files)}"
        saved_dir = os.path.abspath(os.path.join(model_folder, files[0]))

        subfiles = list(os.listdir(saved_dir))
        model_files = [f for f in subfiles if f.endswith(".pt")]
        if model_files:
            model_file = os.path.join(saved_dir, model_files[0])
            self.logger.debug(f"Found model file {model_file}")
        else:
            self.logger.debug("No model found. Not an issue if you've specified a model.")
            model_file = None
        ending = "_aggregated.pickle" if self.variant in ["variational", "bayesian"] else ".pickle"
        pred_file = [f for f in subfiles if f.startswith("PRED") and f.endswith(ending)][0]
        return model_file, os.path.join(saved_dir, pred_file)

    def train(self, force_refresh):
        return self.classify(True, force_refresh)

    def predict(self, force_refresh):
        return self.classify(False, force_refresh)

    def get_types(self):
        t = self.get_simulation_dependency().output
        return t["types"]

    def classify(self, training, force_refresh):
        use_photometry = self.options.get("USE_PHOTOMETRY", False)
        model = self.options.get("MODEL")
        model_path = ""
        if not training:
            assert model is not None, "If TRAIN is not specified, you have to point to a model to use"
            for t in self.dependencies:
                if model == t.name:
                    self.logger.debug(f"Found task dependency {t.name} with model file {t.output['model_filename']}")
                    model = t.output["model_filename"]

            model_path = get_output_loc(model)
            self.logger.debug(f"Looking for model in {model_path}")
            assert os.path.exists(model_path), f"Cannot find {model_path}"

        types = self.get_types()
        if types is None:
            types = OrderedDict({"1": "Ia", "0": "unknown", "2": "SNIax", "3": "SNIa-pec", "20": "SNIIP", "21": "SNIIL", "22": "SNIIn", "29": "SNII", "32": "SNIb", "33": "SNIc", "39": "SNIbc", "41": "SLSN-I", "42": "SLSN-II", "43": "SLSN-R", "80": "AGN", "81": "galaxy", "98": "None", "99": "pending"})
        str_types = json.dumps(types)

        sim_dep = self.get_simulation_dependency()
        light_curve_dir = sim_dep.output["photometry_dir"]
        fit = self.get_fit_dependency()
        fit_dir = f"" if fit is None else f"--fits_dir {fit['fitres_dir']}"
        cyclic = "--cyclic" if self.variant in ["vanilla", "variational"] else ""
        variant = f"--model {self.variant}"

        clump = sim_dep.output.get("clump_file")
        if clump is None:
            clump_txt = ""
        else:
            clump_txt = f"--photo_window_files {clump}"

        format_dict = {
            "conda_env": self.conda_env,
            "dump_dir": self.dump_dir,
            "photometry_dir": light_curve_dir,
            "fit_dir": fit_dir,
            "path_to_classifier": self.path_to_classifier,
            "job_name": self.job_base_name,
            "command": "--train_rnn" if training else "--validate_rnn",
            "sntypes": str_types,
            "variant": variant,
            "cyclic": cyclic,
            "model": "" if training else f"--model_files {model_path}",
            "phot": "" if not use_photometry else "--source_data photometry",
            "test_or_train": "" if training else "--data_testing",
            "done_file": self.done_file,
            "clump": clump_txt
        }

        slurm_output_file = self.output_dir + "/job.slurm"
        self.logger.info(f"Running SuperNNova, slurm job outputting to {slurm_output_file}")
        slurm_text = self.slurm.format(**format_dict)

        old_hash = self.get_old_hash()
        new_hash = self.get_hash_from_string(slurm_text)

        if not force_refresh and new_hash == old_hash:
            self.logger.info("Hash check passed, not rerunning")
        else:
            self.logger.info("Rerunning. Cleaning output_dir")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)

            with open(slurm_output_file, "w") as f:
                f.write(slurm_text)

            self.logger.info(f"Submitting batch job to {'train' if training else 'predict using'} SuperNNova")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)

        return True

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.info("Job complete")

            new_pred_file = self.output_dir + "/predictions.csv"
            new_model_file = self.output_dir + "/model.pt"

            if not os.path.exists(new_pred_file) or not os.path.exists(new_model_file):
                self.logger.info("Updating model location or generating predictions file")
                model, predictions = self.get_model_and_pred()

                if not os.path.exists(new_model_file):
                    if model is not None:
                        shutil.move(model, new_model_file)
                        args_old, args_new = os.path.abspath(os.path.join(os.path.dirname(model), "cli_args.json")), self.output_dir + "/cli_args.json"
                        norm_old, norm_new = os.path.abspath(os.path.join(os.path.dirname(model), "data_norm.json")), self.output_dir + "/data_norm.json"
                        shutil.move(args_old, args_new)
                        shutil.move(norm_old, norm_new)
                        self.logger.info(f"Model file can be found at {new_model_file}")
                if not os.path.exists(new_pred_file):
                    with open(predictions, "rb") as f:
                        dataframe = pickle.load(f)
                        if self.variant in ["variational", "bayesian"]:
                            final_dataframe = dataframe[["SNID", "all_class0_median", "all_class0_std"]]
                            final_dataframe = final_dataframe.rename(columns={
                                "all_class0_median": self.get_prob_column_name(),
                                "all_class0_std": self.get_prob_column_name() + "_ERR",
                            })
                        else:
                            final_dataframe = dataframe[["SNID", "all_class0"]]
                            final_dataframe = final_dataframe.rename(columns={
                                "all_class0": self.get_prob_column_name()
                            })
                        final_dataframe.to_csv(new_pred_file, index=False, float_format="%0.4f")
                        self.logger.info(f"Predictions file can be found at {new_pred_file}")
                chown_dir(self.output_dir)

            self.output.update({
                "model_filename": new_model_file,
                "predictions_filename": new_pred_file
            })
            return Task.FINISHED_SUCCESS
        else:
            num_jobs = self.num_jobs if squeue is None else len([i for i in squeue if self.job_base_name in i])
            if squeue is not None and num_jobs == 0:
                self.logger.warning("SuperNNova has no done file and has no active jobs. This is not good.")
                if os.path.exists(self.hash_file):
                    self.logger.info("Removing hash on failure")
                    os.remove(self.hash_file)
                return Task.FINISHED_FAILURE
            return num_jobs

    @staticmethod
    def get_requirements(options):
        return True, not options.get("USE_PHOTOMETRY", False)