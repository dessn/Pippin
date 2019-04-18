import os
import subprocess
import json
import collections
import shutil
import pickle
from pippin.classifiers.classifier import Classifier
from pippin.config import chown_dir, mkdirs, get_config, get_output_loc
from pippin.task import Task


class SuperNNovaClassifier(Classifier):

    @staticmethod
    def get_requirements(options):
        return True, not options.get("USE_PHOTOMETRY", False)

    def __init__(self, name, light_curve_dir, fit_dir, output_dir, mode, options):
        super().__init__(name, light_curve_dir, fit_dir, output_dir, mode, options)
        self.global_config = get_config()
        self.dump_dir = output_dir + "/dump"
        self.job_base_name = os.path.basename(output_dir)

        self.tmp_output = None
        self.done_file = os.path.join(self.output_dir, "done_task.txt")
        self.slurm = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --account=pi-rkessler
#SBATCH --mem=64GB

source activate {conda_env}
module load cuda
echo `which python`
cd {path_to_classifier}
python run.py --data --sntypes '{sntypes}' --dump_dir {dump_dir} --raw_dir {photometry_dir} --fits_dir {fit_dir} {phot} {test_or_train}
python run.py --use_cuda --cyclic --sntypes '{sntypes}' --done_file {done_file} --dump_dir {dump_dir} {model} {phot} {command}
        """
        self.conda_env = self.global_config["SuperNNova"]["conda_env"]
        self.path_to_classifier = get_output_loc(self.global_config["SuperNNova"]["location"])

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

    def train(self):
        return self.classify(True)

    def predict(self):
        return self.classify(False)

    def classify(self, training):
        use_photometry = self.options.get("USE_PHOTOMETRY", False)
        model = self.options.get("MODEL")
        model_path = ""
        if not training:
            assert model is not None, "If TRAIN is not specified, you have to point to a model to use"
            for t in self.dependencies:
                if model == t.name:
                    self.logger.debug(f"Found task dependency {t.name} with model file {t.output['model']}")
                    model = t.output["model"]

            model_path = get_output_loc(model)
            self.logger.debug(f"Looking for model in {model_path}")
            assert os.path.exists(model_path), f"Cannot find {model_path}"

        types = self.get_types()
        str_types = json.dumps(types)
        format_dict = {
            "conda_env": self.conda_env,
            "dump_dir": self.dump_dir,
            "photometry_dir": self.light_curve_dir,
            "fit_dir": self.fit_dir,
            "path_to_classifier": self.path_to_classifier,
            "job_name": self.job_base_name,
            "command": "--train_rnn" if training else "--validate_rnn",
            "sntypes": str_types,
            "model": "" if training else f"--model_files {model_path}",
            "phot": "" if not use_photometry else "--source_data photometry",
            "test_or_train": "" if training else "--data_testing",
            "done_file": self.output_dir
        }

        slurm_output_file = self.output_dir + "/job.slurm"
        self.logger.info(f"Running SuperNNova, slurm job outputting to {slurm_output_file}")
        slurm_text = self.slurm.format(**format_dict)

        old_hash = self.get_old_hash()
        new_hash = self.get_hash_from_string(slurm_text)

        if new_hash == old_hash:
            self.logger.info("Hash check passed, not rerunning")
        else:
            self.logger.info("Hash check failed, rerunning. Cleaning output_dir")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)

            with open(slurm_output_file, "w") as f:
                f.write(slurm_text)

            self.logger.info("Submitting batch job to train SuperNNova")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)

        return True

    def check_completion(self):
        if os.path.exists(self.done_file):
            self.logger.info("Batch job finished")

            model, predictions = self.get_model_and_pred()
            new_pred_file = self.output_dir + "/predictions.csv"
            new_model_file = self.output_dir + "/model.pt"

            if model is not None:
                shutil.move(model, new_model_file)
                args_old, args_new = os.path.abspath(
                    os.path.join(os.path.dirname(model), "cli_args.json")), self.output_dir + "/cli_args.json"
                shutil.move(args_old, args_new)
                self.logger.info(f"Model file can be found at {new_model_file}")

            with open(predictions, "rb") as f:
                dataframe = pickle.load(f)
                final_dataframe = dataframe[["SNID", "all_class0"]]
                final_dataframe.to_csv(new_pred_file, index=False, float_format="0.4f")
                self.logger.info(f"Predictions file can be found at {new_pred_file}")

            chown_dir(self.output_dir)

            self.output = {"model": new_model_file, "predictions": new_pred_file}

            return Task.FINISHED_GOOD
        else:
            num_jobs = int(subprocess.check_output(f"squeue -h -u $USER -o '%.70j' | grep {self.job_base_name} | wc -l", shell=True))
            if num_jobs == 0:
                if os.path.exists(self.hash_file):
                    self.logger.info("Removing hash on failure")
                    os.remove(self.hash_file)
                return Task.FINISHED_CRASH
            return num_jobs

