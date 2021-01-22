import os
import shutil
import subprocess
import pandas as pd
from pathlib import Path

from pippin.classifiers.classifier import Classifier
from pippin.config import get_config, get_output_loc, mkdirs
from pippin.task import Task


class SnirfClassifier(Classifier):
    """ SNIRF classifier

    CONFIGURATION:
    ==============
    CLASSIFICATION:
      label:
        MASK: TEST  # partial match on sim and classifier
        MASK_SIM: TEST  # partial match on sim name
        MASK_FIT: TEST  # partial match on lcfit name
        MODE: train/predict
        OPTS:
          FITOPT: someLabel # Exact match to fitopt in a fitopt file. USED FOR TRAINING ONLY
          FEATURES: x1 c zHD  # Columns out of fitres file to use as features
          MODEL: someName # exact name of training classification task
          N_ESTIMATORS: 100  # Number of trees in forest
          MIN_SAMPLES_SPLIT: 5  # Min number of samples to split a node on
          MIN_SAMPLES_LEAF: 1  # Minimum number samples in leaf node
          MAX_DEPTH: 0  # Max depth of tree. 0 means auto, which means as deep as it wants.

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        prob_column_name: name of the column to get probabilities out of
        predictions_filename: location of csv filename with id/probs

    """

    def __init__(self, name, output_dir, config, dependencies, mode, options, index=0, model_name=None):
        super().__init__(name, output_dir, config, dependencies, mode, options, index=index, model_name=model_name)
        self.global_config = get_config()
        self.num_jobs = 4

        self.conda_env = self.global_config["SNIRF"]["conda_env"]
        self.path_to_classifier = get_output_loc(self.global_config["SNIRF"]["location"])
        self.job_base_name = os.path.basename(Path(output_dir).parents[1]) + "__" + os.path.basename(output_dir)
        self.features = options.get("FEATURES", "x1 c zHD x1ERR cERR PKMJDERR")
        self.validate_model()

        self.model_pk_file = "model.pkl"
        self.output_pk_file = os.path.join(self.output_dir, self.model_pk_file)
        self.fitopt = options.get("FITOPT", "DEFAULT")
        self.fitres_filename = None
        self.fitres_file = None

        
        self.slurm = """{sbatch_header}
source activate {conda_env}
echo `which python`
cd {path_to_classifier}
python SNIRF.py {command_opts}
"""

    def setup(self):
        lcfit = self.get_fit_dependency()
        self.fitres_filename = lcfit["fitopt_map"][self.fitopt]
        self.fitres_file = os.path.abspath(os.path.join(lcfit["fitres_dirs"][self.index], self.fitres_filename))

    def classify(self, command):
        if self.gpu:
            self.sbatch_header = self.sbatch_gpu_header
        else:
            self.sbatch_header = self.sbatch_cpu_header

        header_dict = {
                    "job-name": self.job_base_name,
                    "output": "output.log",
                    "time": "15:00:00",
                    "mem-per-cpu": "3GB",
                    "ntasks": "1",
                    "cpus-per-task": "4"
                }

        self.update_header(header_dict)

        format_dict = {
            "sbatch_header": self.sbatch_header,
            "conda_env": self.conda_env,
            "path_to_classifier": self.path_to_classifier,
            "command_opts": command,
            "done_file": self.done_file,
        }
        slurm_script = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string(slurm_script)

        if self._check_regenerate(new_hash):
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
            self.logger.info("Hash check passed, not rerunning")
            self.should_be_done()
        return True

    def get_rf_conf(self):
        leaf_opts = (
            (f"--n_estimators {self.options.get('N_ESTIMATORS')} " if self.options.get("N_ESTIMATORS") is not None else "")
            + (f"--min_samples_split {self.options.get('MIN_SAMPLES_SPLIT')} " if self.options.get("MIN_SAMPLES_SPLIT") is not None else "")
            + (f"--min_samples_leaf {self.options.get('MIN_SAMPLES_LEAF')} " if self.options.get("MIN_SAMPLES_LEAF") is not None else "")
            + (f"--max_depth {self.options.get('MAX_DEPTH')} " if self.options.get("MAX_DEPTH") is not None else "")
        )
        return leaf_opts

    def predict(self):
        self.setup()
        model = self.options.get("MODEL")
        if model is None:
            self.logger.error("If you are in predict model, please specify a MODEL in OPTS. Either a file location or a training task name.")
            return False
        potential_path = get_output_loc(model)
        if os.path.exists(potential_path):
            self.logger.debug(f"Found existing model file at {potential_path}")
            model = potential_path
        else:
            if "/" in model:
                self.logger.warning(f"Your model {model} looks like a path, but I couldn't find a model at {potential_path}")
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
            f"{self.get_rf_conf()}"
            f"--test {self.fitres_file} "
            f"--filedir {self.output_dir} "
            f"--done_file {self.done_file} "
            f"--use_filenames "
        )
        return self.classify(command)

    def train(self):
        self.setup()
        command = (
            f"--nc 4 "
            f"--nclass 2 "
            f"--ft {self.features} "
            f"--train_only "
            f"--test '' "
            f"--pklfile {self.output_pk_file} "
            f"--pklformat FITRES "
            f"{self.get_rf_conf()}"
            f"--filedir {self.output_dir} "
            f"--train {self.fitres_file} "
            f"--done_file {self.done_file} "
            f"--use_filenames "
        )
        return self.classify(command)

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
        return self.check_for_job(squeue, self.job_base_name)

    @staticmethod
    def get_requirements(options):
        return False, True
