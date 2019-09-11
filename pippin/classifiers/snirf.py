import os
import shutil
import subprocess
import pandas as pd

from pippin.classifiers.classifier import Classifier
from pippin.config import get_config, get_output_loc, mkdirs
from pippin.task import Task


class SnirfClassifier(Classifier):
    def __init__(self, name, output_dir, dependencies, mode, options):
        super().__init__(name, output_dir, dependencies, mode, options)
        self.global_config = get_config()
        self.num_jobs = 4

        self.conda_env = self.global_config["ArgonneClassifier"]["conda_env"]
        self.path_to_classifier = get_output_loc(self.global_config["ArgonneClassifier"]["location"])
        self.job_base_name = os.path.basename(output_dir)
        self.features = options.get("FEATURES", "x1 c zHD x1ERR cERR PKMJDERR")
        self.model_pk_file = "modelpkl.pkl"
        self.output_pk_file = os.path.join(self.output_dir, self.model_pk_file)
        self.fitopt = options.get("FITOPT", "DEFAULT")
        lcfit = self.get_fit_dependency()
        self.fitres_filename = lcfit["fitopt_map"][self.fitopt]
        self.fitres_file = os.path.abspath(os.path.join(lcfit["fitres_dir"], self.fitres_filename))

        self.slurm = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=broadwl
#SBATCH --output=output.log
#SBATCH --account=pi-rkessler
#SBATCH --mem=14GB

source activate {conda_env}
echo `which python`
cd {path_to_classifier}
python SNIRF.py {command_opts}
"""

    def classify(self, force_refresh, command):
        format_dict = {"job_name": self.job_base_name, "conda_env": self.conda_env, "path_to_classifier": self.path_to_classifier, "command_opts": command}
        slurm_script = self.slurm.format(**format_dict)

        old_hash = self.get_old_hash()
        new_hash = self.get_hash_from_string(slurm_script)

        if force_refresh or new_hash != old_hash:
            self.logger.debug("Regenerating")

            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)

            slurm_output_file = self.output_dir + "/job.slurm"
            with open(slurm_output_file, "w") as f:
                f.write(slurm_script)
            self.save_new_hash(new_hash)
            self.logger.info(f"Submitting batch job {slurm_output_file}")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)
        else:
            self.logger.debug("Not regenerating")
        return True

    def predict(self, force_refresh):
        model = self.options.get("MODEL")
        if model is None:
            self.logger.error("If you are in predict model, please specify a MODEL in OPTS. Either a file location or a training task name.")
            return False
        if not os.path.exists(get_output_loc(model)):
            # If its not a file, it must be a task
            for t in self.dependencies:
                if model == t.name:
                    self.logger.debug(f"Found task dependency {t.name} with model file {t.output['model_filename']}")
                    model = t.output["model_filename"]
        command = (
            f"--nc 4 "
            f"--nclass 2 "
            f"--ft {self.features} "
            f"--restore "
            f"--pklfile {model} "
            f"--pklformat FITRES "
            f"--test {self.fitres_file} "
            f"--filedir {self.output_dir} "
            f"--done_file {self.done_file} "
            f"--use_filenames "
        )
        return self.classify(force_refresh, command)

    def train(self, force_refresh):
        command = (
            f"--nc 4 "
            f"--nclass 2 "
            f"--ft {self.features} "
            f"--train_only "
            f"--test '' "
            f"--pklfile {self.output_pk_file} "
            f"--pklformat FITRES "
            f"--filedir {self.output_dir} "
            f"--train {self.fitres_file} "
            f"--done_file {self.done_file} "
            f"--use_filenames "
        )
        return self.classify(force_refresh, command)

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Found done file at {self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read().upper():
                    return Task.FINISHED_FAILURE
                else:
                    if self.mode == Classifier.PREDICT:
                        # Rename output file myself
                        # First check to see if this is already done
                        predictions_filename = os.path.join(self.output_dir, "predictions.csv")
                        if not os.path.exists(predictions_filename):
                            # Find the output file
                            output_files = [i for i in os.listdir(self.output_dir) if i.endswith("Classes.txt")]
                            if len(output_files) != 1:
                                self.logger.error(f"Could not find the output file in {self.output_dir}")
                                return Task.FINISHED_FAILURE
                            df = pd.read_csv(os.path.join(self.output_dir, output_files[0]), delim_whitespace=True)
                            df_final = df[["CID", "RFprobability0"]]
                            df_final = df_final.rename(columns={"CID": "SNID", "RFprobability0": self.get_prob_column_name()})
                            df_final.to_csv(predictions_filename, index=False, float_format="%0.4f")
                        self.output["predictions_filename"] = predictions_filename
                    else:
                        self.output["model_filename"] = [
                            os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.startswith(self.model_pk_file)
                        ][0]
                    return Task.FINISHED_SUCCESS
        return self.num_jobs

    @staticmethod
    def get_requirements(options):
        return False, True
