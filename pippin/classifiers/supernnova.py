import os
import inspect
import subprocess
import json
import collections
import shutil
import pickle
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs, get_config


class SuperNNovaClassifier(Classifier):
    def __init__(self, light_curve_dir, fit_dir, output_dir, options):
        super().__init__(light_curve_dir, fit_dir, output_dir, options)
        self.global_config = get_config()
        self.dump_dir = output_dir + "/dump"
        self.job_base_name = os.path.basename(output_dir)

        self.slurm = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --account=pi-rkessler
#SBATCH --mem=16G

source ~/.bashrc
conda activate {conda_env}
module load cuda
cd {path_to_supernnova}
python run.py --data --sntypes '{sntypes}' --dump_dir {dump_dir} --raw_dir {photometry_dir} --fits_dir {fit_dir} {test_or_train}
python run.py --use_cuda --sntypes '{sntypes}' --dump_dir {dump_dir} {model} {command}
        """
        self.conda_env = self.global_config["SuperNNova"]["conda_env"]
        self.path_to_supernnova = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../../../" + self.global_config["SuperNNova"]["location"])

    def get_types(self):
        types = {}
        sim_config_dir = os.path.abspath(os.path.join(self.light_curve_dir, os.pardir))
        self.logger.debug(f"Searching {sim_config_dir} for types")
        for f in [f for f in os.listdir(sim_config_dir) if f.endswith(".input")]:
            path = os.path.join(sim_config_dir, f)
            name = f.split(".")[0]
            with open(path, "r") as file:
                for line in file.readlines():
                    if line.startswith("GENTYPE"):
                        number = "1" + "%02d" % int(line.split(":")[1].strip())
                        types[number] = name
                        break
        sorted_types = collections.OrderedDict(sorted(types.items()))
        self.logger.info(f"Types found: {json.dumps(sorted_types)}")
        return sorted_types

    def get_model_and_pred(self):
        model_folder = self.dump_dir + "/models"
        files = [f for f in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, f))]
        assert len(files) == 1, f"More than one directory found: {str(files)}"
        saved_dir = os.path.abspath(os.path.join(model_folder, files[0]))

        subfiles = list(os.listdir(saved_dir))
        model_files = [f for f in subfiles if f.endswith(".pt")]
        if model_files:
            model_file = os.path.join(saved_dir, model_files[0])
            self.logger.debug(f"Found model file {model_file}")
        else:
            self.logger.debug("No model found. Not an issue if you've specified a model.")
            model_file = None
        pred_file = [f for f in subfiles if f.startswith("PRED") and f.endswith(".pickle")][0]
        return model_file, os.path.join(saved_dir, pred_file)

    def classify(self):
        if os.path.exists(self.output_dir):
            self.logger.info(f"Removing old output files at {self.output_dir}")
            shutil.rmtree(self.output_dir)
        mkdirs(self.output_dir)

        training = self.options.get("TRAIN") is not None
        model = self.options.get("MODEL")
        model_path = None
        if not training:
            assert model is not None, "If TRAIN is not specified, you have to point to a model to use"
            model_path = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../../" + model)
            self.logger.debug(f"Looking for model in {model_path}")
            assert os.path.exists(model_path), f"Cannot find {model_path}"

        types = self.get_types()
        str_types = json.dumps(types)
        format_dict = {
            "conda_env": self.conda_env,
            "dump_dir": self.dump_dir,
            "photometry_dir": self.light_curve_dir,
            "fit_dir": self.fit_dir,
            "path_to_supernnova": self.path_to_supernnova,
            "job_name": f"train_{self.job_base_name}",
            "command": "--train_rnn" if training else "--validate_rnn",
            "sntypes": str_types,
            "model": "" if training else f"--model_files {model_path}" ,
            "test_or_train": "" if training else "--data_testing"
        }

        slurm_output_file = self.output_dir + "/job.slurm"
        self.logger.info(f"Running SuperNNova, slurm job outputting to {slurm_output_file}")

        with open(slurm_output_file, "w") as f:
            f.write(self.slurm.format(**format_dict))

        self.logger.info("Submitting batch job to train SuperNNova")
        subprocess.run(["sbatch", "--wait", slurm_output_file], cwd=self.output_dir)
        self.logger.info("Batch job finished")

        model, predictions = self.get_model_and_pred()
        new_pred_file = self.output_dir + "/predictions.csv"
        new_model_file = self.output_dir + "/model.pt"

        if model is not None:
            shutil.move(model, new_model_file)
            args_old, args_new = os.path.abspath(os.path.join(os.path.dirname(model), "cli_args.json")), self.output_dir + "/cli_args.json"
            shutil.move(args_old, args_new)
            self.logger.info(f"Model file can be found at {new_model_file}")

        with open(predictions, "rb") as f:
            dataframe = pickle.load(f)
            final_dataframe = dataframe[["SNID", "all_class0"]]
            final_dataframe.to_csv(new_pred_file)
            self.logger.info(f"Predictions file can be found at {new_pred_file}")

        chown_dir(self.output_dir)
        return True  # change to hash
