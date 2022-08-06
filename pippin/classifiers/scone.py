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
    """ convolutional neural network-based SN photometric classifier
    for details, see https://arxiv.org/abs/2106.04370, https://arxiv.org/abs/2111.05539, https://arxiv.org/abs/2207.09440

    CONFIGURATION:
    ==============
    CLASSIFICATION:
      label:
        CLASSIFIER: SconeClassifier
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
          MODEL: /path/to/trained/model
          SCONE_CPU_BATCH_FILE: /path/to/sbatch/template/for/scone
          SCONE_GPU_BATCH_FILE: /path/to/sbatch/template/for/scone
          BATCH_REPLACE: {}

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
      self.init_env_heatmaps = self.global_config["SCONE"]["init_env_cpu"]
      self.init_env = self.global_config["SCONE"]["init_env_cpu"] if not self.gpu else self.global_config["SCONE"]["init_env_gpu"]
      self.path_to_classifier = self.global_config["SCONE"]["location"]

      output_path_obj = Path(self.output_dir)
      heatmaps_path_obj = output_path_obj / "heatmaps"

      self.job_base_name = output_path_obj.parents[1].name + "__" + output_path_obj.name
      self.heatmaps_job_base_name = self.job_base_name + "__CREATE_HEATMAPS"

      self.batch_replace = self.options.get("BATCH_REPLACE", {})
      self.slurm = """{sbatch_header}
      {task_setup}"""

      self.config_path = str(output_path_obj / "model_config.yml")
      self.heatmaps_path = str(heatmaps_path_obj)
      self.heatmaps_done_file = str(heatmaps_path_obj / "done.txt")
      self.heatmaps_sbatch_header_path = str(heatmaps_path_obj / "sbatch_header.sh")

      self.logfile = str(output_path_obj / "output.log")

      remake_heatmaps = self.options.get("REMAKE_HEATMAPS", False)
      self.keep_heatmaps = not remake_heatmaps

    def make_sbatch_header(self, option_name, header_dict, use_gpu=False):
      sbatch_header_template = self.options.get(option_name)
      sbatch_header = self.sbatch_cpu_header

      if sbatch_header_template is not None:
        self.logger.debug(f"batch file found at {sbatch_header_template}")
        with open(get_data_loc(sbatch_header_template), 'r') as f:
          sbatch_header = f.read()

      sbatch_header = self.clean_header(sbatch_header)

      header_dict = merge_dict(header_dict, self.batch_replace)
      return self._update_header(sbatch_header, header_dict)

    def make_heatmaps(self, mode):
        self.logger.info("heatmaps not created, creating now")
        shutil.rmtree(self.output_dir, ignore_errors=True)
        mkdirs(self.heatmaps_path)

        sim_dep = self.get_simulation_dependency()
        sim_dirs = sim_dep.output["photometry_dirs"]

        lcdata_paths = self._get_lcdata_paths(sim_dirs)
        metadata_paths = [path.replace("PHOT", "HEAD") for path in lcdata_paths]

        # TODO: if externally specified batchfile exists, have to parse desired logfile path from it
        header_dict = {
              "REPLACE_LOGFILE": self.heatmaps_log_path,
              "REPLACE_WALLTIME": "1:00:00",
              "REPLACE_MEM": "8GB",
              "APPEND": ["#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8"]
            }
        heatmaps_sbatch_header = self.make_sbatch_header("HEATMAPS_BATCH_FILE", header_dict)
        with open(self.heatmaps_sbatch_header_path, "w+") as f:
          f.write(heatmaps_sbatch_header)
        
        self._write_config_file(metadata_paths, lcdata_paths, mode, self.config_path) # TODO: what if they don't want to train on all sims?

        # call create_heatmaps/run.py, which sbatches X create heatmaps jobs
        subprocess.run([f"python {Path(self.path_to_classifier) / 'create_heatmaps/run.py'} --config_path {self.config_path}"], shell=True)

    def classify(self, mode):
      heatmaps_created = self._heatmap_creation_success() and self.keep_heatmaps

      if not heatmaps_created:
        self.heatmaps_log_path = os.path.join(self.heatmaps_path, f"create_heatmaps__{os.path.basename(self.config_path).split('.')[0]}.log")
        self.make_heatmaps(mode)
        # TODO: check status in a different job? but what if the job doesn't run or runs after the other ones are already completed?
        # -- otherwise if ssh connection dies the classification won't run
        # -- any better solution than while loop + sleep?

        start_sleep_time = self.global_config["OUTPUT"]["ping_frequency"]
        max_sleep_time = self.global_config["OUTPUT"]["max_ping_frequency"]
        current_sleep_time = start_sleep_time

        while self.num_jobs_in_queue() > 0:
          self.logger.debug(f"> 0 {self.job_base_name} jobs still in the queue, sleeping for {current_sleep_time}")
          time.sleep(current_sleep_time)
          current_sleep_time *= 2
          if current_sleep_time > max_sleep_time:
            current_sleep_time = max_sleep_time

        self.logger.debug("jobs done, evaluating success")
        if not self._heatmap_creation_success():
          self.logger.error(f"heatmaps were not created successfully, see logs at {self.heatmaps_log_path}")
          return Task.FINISHED_FAILURE

      # when all done, sbatch a gpu job for actual classification
      self.logger.info("heatmaps created, continuing")
      
      failed = False
      if os.path.exists(self.done_file):
        self.logger.debug(f"Found done file at {self.done_file}")
        with open(self.done_file) as f:
          if "FAILURE" in f.read().upper():
            failed = True

      header_dict = {
              "REPLACE_NAME": self.job_base_name,
              "REPLACE_LOGFILE": "output.log",
              "REPLACE_WALLTIME": "4:00:00", # max for gpu
              "REPLACE_MEM": "8GB",
              "APPEND": ["#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8"]
              }
      model_sbatch_header = self.make_sbatch_header("MODEL_BATCH_FILE", header_dict, use_gpu=self.gpu)

      setup_dict = {
              "init_env": self.init_env,
              "path_to_classifier": self.path_to_classifier,
              "heatmaps_path": self.heatmaps_path,
              "config_path": self.config_path,
              "done_file": self.done_file,
              }

      format_dict = {
              "sbatch_header": model_sbatch_header,
              "task_setup": self.update_setup(setup_dict, self.task_setup['scone'])
              }
      slurm_output_file = Path(self.output_dir) / "job.slurm"
      self.logger.info(f"Running SCONE, slurm job outputting to {slurm_output_file}")
      slurm_script = self.slurm.format(**format_dict)

      new_hash = self.get_hash_from_string(slurm_script)

      if self._check_regenerate(new_hash) or failed:
        self.logger.debug("Regenerating")

        with open(slurm_output_file, "w") as f:
            f.write(slurm_script)
        self.save_new_hash(new_hash)
        self.logger.info(f"Submitting batch job {slurm_output_file}")

        # TODO: nersc needs `module load esslurm` to sbatch gpu jobs, maybe make 
        # this shell command to a file so diff systems can define their own
        subprocess.run(f"sbatch {slurm_output_file}", cwd=self.output_dir, shell=True)
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

        # environment configuration
        config["init_env_heatmaps"] = self.init_env_heatmaps
        config["init_env"] = self.init_env

        # info for heatmap creation
        config["metadata_paths"] = metadata_paths
        config["lcdata_paths"] = lcdata_paths
        config["num_wavelength_bins"] = self.options.get("NUM_WAVELENGTH_BINS", 32)
        config["num_mjd_bins"] = self.options.get("NUM_MJD_BINS", 180)
        config["heatmaps_path"] = self.heatmaps_path
        config["sbatch_header_path"] = self.heatmaps_sbatch_header_path
        config["heatmaps_donefile"] = self.heatmaps_done_file
        config["heatmaps_logfile"] = self.heatmaps_log_path
        config["sim_fraction"] = self.options.get("SIM_FRACTION", 1) # 1/sim_fraction % of simulated SNe will be used for the model

        # info for classification model
        config["categorical"] = self.options.get("CATEGORICAL", False)
        config["num_epochs"] = self.options.get("NUM_EPOCHS", 400) # TODO: replace num epochs with autostop: stop training when slope plateaus?
        config["Ia_fraction"] = self.options.get("IA_FRACTION", 0.5)
        config["output_path"] = self.output_dir
        config["trained_model"] = self.options.get("MODEL", False)
        config["kcor_file"] = self.options.get("KCOR_FILE", None)
        config["mode"] = mode
        config["job_base_name"] = self.heatmaps_job_base_name
        #FOR DES DATA: 
        # config["sn_type_id_to_name"] ={0.0: "unknown",
        #    5.0: "non",
        #    23.0: "non",
        #    29.0: "non",
        #    32.0: "non",
        #    33.0: "non",
        #    39.0: "non",
        #    41.0: "non",
        #    42.0: "non",
        #    64.0: "non",
        #    66.0: "non",
        #    67: "non",
        #    80.0: "non",
        #    81.0: "non",
        #    82.0: "non",
        #    90.0: "SNIa",
        #    129.0: "non",
        #    139.0: "non",
        #    141.0: "non",
        #    180.0: "non"}

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
            self.output.update({"model_filename": self.options.get("MODEL", os.path.join(self.output_dir, "trained_model")), "predictions_filename": pred_path})
            return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.heatmaps_job_base_name) + self.check_for_job(squeue, self.job_base_name)

    def _heatmap_creation_success(self):
        if not os.path.exists(self.heatmaps_done_file):
            return False
        with open(self.heatmaps_done_file, "r") as donefile:
            if "CREATE HEATMAPS FAILURE" in donefile.read():
                return False
        return os.path.exists(self.heatmaps_path) and os.path.exists(os.path.join(self.heatmaps_path, "done.log")) 
    
    def num_jobs_in_queue(self):
        print("rerun num jobs in queue")
        squeue = [i.strip() for i in subprocess.check_output(f"squeue -h -u $USER -o '%.200j'", shell=True, text=True).splitlines()]
        self.logger.debug(f"{squeue}")
        return self.check_for_job(squeue, self.job_base_name)

    @staticmethod
    def _get_lcdata_paths(sim_dirs):
        sim_paths = [f.path for sim_dir in sim_dirs for f in os.scandir(sim_dir)]
        lcdata_paths = [path for path in sim_paths if "PHOT" in path]

        return lcdata_paths

    @staticmethod
    def _update_header(header, header_dict):
      for key, value in header_dict.items():
        if key in header:
          header = header.replace(key, str(value))
      append_list = header_dict.get("APPEND")
      if append_list is not None:
        lines = header.split('\n')
        lines += append_list
        header = '\n'.join(lines)
      return header

    @staticmethod
    def get_requirements(options):
        return True, False
