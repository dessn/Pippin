import os
import shutil
import subprocess
from pathlib import Path
import yaml
import pandas as pd
import re
import numpy as np
import time

from pippin.classifiers.classifier import Classifier
from pippin.config import get_config, get_output_loc, mkdirs, get_data_loc, merge_dict
from pippin.task import Task

class SconeClassifier(Classifier):
    """ Nearest Neighbor Python classifier

    CONFIGURATION:
    ==============
    CLASSIFICATION:
      label:
        MASK: TEST  # partial match on sim and classifier
        MASK_SIM: TEST  # partial match on sim name
        MASK_FIT: TEST  # partial match on lcfit name
        MODE: train/predict
        OPTS:
          GPU: True
          CATEGORICAL: False
          NUM_WAVELENGTH_BINS: 32
          NUM_MJD_BINS: 180
          REMAKE_HEATMAPS: False 
          NUM_EPOCHS: 400 
          IA_FRACTION: 0.5
          TRAINED_MODEL: /path/to/trained/model

    OUTPUTS:
    ========
      predictions.csv: list of snids and associated predictions
      training_history.csv: training history output from keras

    """

    def __init__(self, name, output_dir, config, dependencies, mode, options, index=0, model_name=None):
        super().__init__(name, output_dir, config, dependencies, mode, options, index=index, model_name=model_name)
        self.global_config = get_config()
        self.options = options

        self.gpu = self.options.get("GPU", True)
        self.conda_env = self.global_config["SCONE"]["conda_env_cpu"] if not self.gpu else self.global_config["SCONE"]["conda_env_gpu"]
        self.path_to_classifier = self.global_config["SCONE"]["location"]

        self.job_base_name = os.path.basename(Path(output_dir).parents[1]) + "__" + os.path.basename(output_dir)

        self.batch_file = self.options.get("BATCH_FILE")
        if self.batch_file is not None:
            self.batch_file = get_data_loc(self.batch_file)
        self.batch_replace = self.options.get("BATCH_REPLACE", {})


        self.config_path = os.path.join(self.output_dir, "model_config.yml")
        self.heatmaps_path = os.path.join(self.output_dir, "heatmaps")
        self.csvs_path = os.path.join(self.output_dir, "sim_csvs")
        self.slurm = """{sbatch_header}
        {task_setup}"""

        self.logfile = os.path.join(self.output_dir, "output.log")

        remake_heatmaps = self.options.get("REMAKE_HEATMAPS", False)
        self.keep_heatmaps = not remake_heatmaps

    def classify(self, mode):
        header_dict = {
                "REPLACE_NAME": self.job_base_name,
                "REPLACE_LOGFILE": "output.log",
                "REPLACE_WALLTIME": "15:00:00", # TODO: scale based on number of heatmaps
                "REPLACE_MEM": "8GB",
                # "APPEND": ["#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8"]
                }
        header_dict = merge_dict(header_dict, self.batch_replace)
        if self.batch_file is None:
            if self.gpu:
                self.sbatch_header = self.sbatch_gpu_header
            else:
                self.sbatch_header = self.sbatch_cpu_header
        else:
            with open(self.batch_file, 'r') as f:
                self.sbatch_header = f.read()
            self.sbatch_header = self.clean_header(self.sbatch_header)

        self.update_header(header_dict)

        setup_dict = {
                "conda_env": self.conda_env,
                "path_to_classifier": self.path_to_classifier,
                "heatmaps_path": self.heatmaps_path,
                "config_path": self.config_path,
                "done_file": self.done_file,
                }

        format_dict = {
                "sbatch_header": self.sbatch_header,
                "task_setup": self.update_setup(setup_dict, self.task_setup['scone'])
                }
        slurm_output_file = self.output_dir + "/job.slurm"
        self.logger.info(f"Running SCONE, slurm job outputting to {slurm_output_file}")
        slurm_script = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string(slurm_script)

        # check success of intermediate steps and don't redo them if successful
        heatmaps_created = self._heatmap_creation_success() and self.keep_heatmaps

        failed = False
        if os.path.exists(self.done_file):
            self.logger.debug(f"Found done file at {self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read().upper():
                    failed = True

        if self._check_regenerate(new_hash) or failed:
            self.logger.debug("Regenerating")
            if not heatmaps_created:
                shutil.rmtree(self.output_dir, ignore_errors=True)
                mkdirs(self.output_dir)
            else:
                for f in [f.path for f in os.scandir(self.output_dir) if f.is_file()]:
                    os.remove(f)

            sim_dep = self.get_simulation_dependency()
            sim_dirs = sim_dep.output["photometry_dirs"]

            lcdata_paths = self._get_lcdata_paths(sim_dirs)
            metadata_paths = [path.replace("PHOT", "HEAD") for path in lcdata_paths]
            self._write_config_file(metadata_paths, lcdata_paths, mode, self.config_path) # TODO: what if they don't want to train on all sims?

            with open(slurm_output_file, "w") as f:
                f.write(slurm_script)
            self.save_new_hash(new_hash)
            self.logger.info(f"Submitting batch job {slurm_output_file}")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")
            self.should_be_done()
        return True

    def predict(self):
        return self.classify("predict")

    def train(self):
        return self.classify("train")

   #TODO: investigate the output and use this 
    def _get_types(self):
        t = self.get_simulation_dependency().output
        return t["types"]

    def _write_config_file(self, metadata_paths, lcdata_paths, mode, config_path):
        config = {}
        config["categorical"] = self.options.get("CATEGORICAL", False)
        # TODO: replace num epochs with autostop: stop training when slope plateaus?
        # TODO: how to choose optimal batch size?
        config["num_epochs"] = self.options.get("NUM_EPOCHS")
        config["metadata_paths"] = metadata_paths
        config["lcdata_paths"] = lcdata_paths
        config["heatmaps_path"] = self.heatmaps_path
        config["num_wavelength_bins"] = self.options.get("NUM_WAVELENGTH_BINS", 32)
        config["num_mjd_bins"] = self.options.get("NUM_MJD_BINS", 180)
        config["Ia_fraction"] = self.options.get("IA_FRACTION", 0.5)
        config["donefile"] = self.done_file
        config["output_path"] = self.output_dir
        config["trained_model"] = self.options.get("TRAINED_MODEL", False)
        config["kcor_file"] = self.options.get("KCOR_FILE", None)
        config["mode"] = mode
        known_types = {42: "SNII",
          52: "SNIax",
          62: "SNIbc",
          67: "SNIa-91bg",
          64: "KN",
          90: "SNIa",
          95: "SLSN-1"}
        # config["sn_type_id_to_name"] = {int(k):known_types[k] if k in known_types else "non" for k in types}
        #FOR DES DATA: 
        config["sn_type_id_to_name"] ={0.0: "unknown",
           5.0: "non",
           23.0: "non",
           29.0: "non",
           32.0: "non",
           33.0: "non",
           39.0: "non",
           41.0: "non",
           42.0: "non",
           64.0: "non",
           66.0: "non",
           67: "non",
           80.0: "non",
           81.0: "non",
           82.0: "non",
           90.0: "SNIa",
           129.0: "non",
           139.0: "non",
           141.0: "non",
           180.0: "non"}

        with open(config_path, "w+") as cfgfile:
            cfgfile.write(yaml.dump(config))

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Found done file at {self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read().upper():
                    return Task.FINISHED_FAILURE

            time.sleep(30)
            pred_path = os.path.join(self.output_dir, "predictions.csv")
            predictions = pd.read_csv(pred_path)
            predictions = predictions.rename(columns={"pred": self.get_prob_column_name()})
            predictions.to_csv(pred_path, index=False)
            self.logger.info(f"Predictions file can be found at {pred_path}")
            self.output.update({"model_filename": self.options.get("TRAINED_MODEL", os.path.join(self.output_dir, "trained_model")), "predictions_filename": pred_path})
            return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_base_name)

    def _heatmap_creation_success(self):
        if not os.path.exists(self.done_file):
            return False
        with open(self.done_file, "r") as donefile:
            if "CREATE HEATMAPS FAILURE" in donefile.read():
                return False
        return os.path.exists(self.heatmaps_path) and os.path.exists(os.path.join(self.heatmaps_path, "done.log")) 
        
    @staticmethod
    def _get_lcdata_paths(sim_dirs):
        sim_paths = [f.path for sim_dir in sim_dirs for f in os.scandir(sim_dir)]
        lcdata_paths = [path for path in sim_paths if "PHOT" in path]

        return lcdata_paths

    @staticmethod
    def get_requirements(options):
        return True, False
