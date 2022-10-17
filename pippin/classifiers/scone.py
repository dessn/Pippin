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

      self.batch_replace = self.options.get("BATCH_REPLACE", {})
      self.slurm = """{sbatch_header}
      {task_setup}"""

      self.config_path = str(output_path_obj / "model_config.yml")
      self.logfile = str(output_path_obj / "output.log")
      self.model_sbatch_job_path = str(output_path_obj / "job.slurm")

      self.heatmaps_path = str(heatmaps_path_obj)
      self.heatmaps_done_file = str(heatmaps_path_obj / "done.txt")
      self.heatmaps_sbatch_header_path = str(heatmaps_path_obj / "sbatch_header.sh")
      self.heatmaps_log_path = str(heatmaps_path_obj / f"create_heatmaps__{Path(self.config_path).name.split('.')[0]}.log")

      remake_heatmaps = self.options.get("REMAKE_HEATMAPS", False)
      self.keep_heatmaps = not remake_heatmaps

    def make_sbatch_header(self, option_name, header_dict, use_gpu=False):
      sbatch_header_template = self.options.get(option_name)
      sbatch_header = self.sbatch_gpu_header if use_gpu else self.sbatch_cpu_header

      if sbatch_header_template is not None:
        self.logger.debug(f"batch file found at {sbatch_header_template}")
        with open(get_data_loc(sbatch_header_template), 'r') as f:
          sbatch_header = f.read()

      sbatch_header = self.clean_header(sbatch_header)

      header_dict = merge_dict(header_dict, self.batch_replace)
      return self._update_header(sbatch_header, header_dict)

    def make_heatmaps_sbatch_header(self):
      self.logger.info("heatmaps not created, creating now")
      shutil.rmtree(self.output_dir, ignore_errors=True)
      mkdirs(self.heatmaps_path)

      # TODO: if externally specified batchfile exists, have to parse desired logfile path from it
      header_dict = {
            "REPLACE_LOGFILE": self.heatmaps_log_path,
            "REPLACE_WALLTIME": "10:00:00", #TODO: change to scale with # of heatmaps expected
            "REPLACE_MEM": "16GB",
          }
      heatmaps_sbatch_header = self.make_sbatch_header("HEATMAPS_BATCH_FILE", header_dict)

      with open(self.heatmaps_sbatch_header_path, "w+") as f:
        f.write(heatmaps_sbatch_header)

    def make_model_sbatch_script(self):
      header_dict = {
              "REPLACE_NAME": self.job_base_name,
              "REPLACE_LOGFILE": str(Path(self.output_dir) / "output.log"),
              "REPLACE_MEM": "16GB",
              "REPLACE_WALLTIME": "4:00:00" if self.gpu else "12:00:00", # 4h is max for gpu
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

      self.logger.info(f"Running SCONE model, slurm job written to {self.model_sbatch_job_path}")
      slurm_script = self.slurm.format(**format_dict)

      with open(self.model_sbatch_job_path, "w") as f:
          f.write(slurm_script)

      return slurm_script

    def classify(self, mode):
      failed = False
      if Path(self.done_file).exists():
          self.logger.debug(f"Found done file at {self.done_file}")
          with open(self.done_file) as f:
            if "SUCCESS" not in f.read().upper():
              failed = True

      heatmaps_created = self._heatmap_creation_success() and self.keep_heatmaps

      sim_dep = self.get_simulation_dependency()
      sim_dirs = sim_dep.output["photometry_dirs"][self.index] # if multiple realizations, get only the current one with self.index

      lcdata_paths = self._get_lcdata_paths(sim_dirs)
      metadata_paths = [path.replace("PHOT", "HEAD") for path in lcdata_paths]

      str_config = self._make_config(metadata_paths, lcdata_paths, mode, heatmaps_created)
      new_hash = self.get_hash_from_string(str_config)

      if self._check_regenerate(new_hash) or failed:
        self.logger.debug("Regenerating")
      else:
        self.logger.info("Hash check passed, not rerunning")
        self.should_be_done()
        return True

      if not heatmaps_created:
        # this deletes the whole directory tree, don't write anything before this
        self.make_heatmaps_sbatch_header()

      self.save_new_hash(new_hash)
      with open(self.config_path, "w+") as cfgfile:
          cfgfile.write(str_config)

      slurm_script = self.make_model_sbatch_script()

      self.logger.info(f"Submitting batch job {self.model_sbatch_job_path}")

      # TODO: nersc needs `module load esslurm` to sbatch gpu jobs, maybe make
      # this shell command to a file so diff systems can define their own
      subprocess.run([f"python {Path(self.path_to_classifier) / 'run.py'} --config_path {self.config_path}"], shell=True)
      return True

    def predict(self):
        return self.classify("predict")

    def train(self):
        return self.classify("train")

   #TODO: investigate the output and use this
    def _get_types(self):
        t = self.get_simulation_dependency().output
        return t["types"]

    def _make_config(self, metadata_paths, lcdata_paths, mode, heatmaps_created):
        config = {}

        # environment configuration
        config["init_env_heatmaps"] = self.init_env_heatmaps
        config["init_env"] = self.init_env

        # info for heatmap creation
        if not heatmaps_created:
          config["sbatch_header_path"] = self.heatmaps_sbatch_header_path

        config["heatmaps_donefile"] = self.heatmaps_done_file
        config["heatmaps_logfile"] = self.heatmaps_log_path
        config["sim_fraction"] = self.options.get("SIM_FRACTION", 1) # 1/sim_fraction % of simulated SNe will be used for the model
        config["heatmaps_path"] = self.heatmaps_path
        config["model_sbatch_job_path"] = self.model_sbatch_job_path
        config["num_wavelength_bins"] = self.options.get("NUM_WAVELENGTH_BINS", 32)
        config["num_mjd_bins"] = self.options.get("NUM_MJD_BINS", 180)
        config["metadata_paths"] = metadata_paths
        config["lcdata_paths"] = lcdata_paths

        # info for classification model
        config["categorical"] = self.options.get("CATEGORICAL", False)
        config["num_epochs"] = self.options.get("NUM_EPOCHS", 400) # TODO: replace num epochs with autostop: stop training when slope plateaus?
        config["batch_size"] = self.options.get("BATCH_SIZE", 32) # TODO: replace with percentage of total size?
        config["Ia_fraction"] = self.options.get("IA_FRACTION", 0.5)
        config["output_path"] = self.output_dir
        config["trained_model"] = self.options.get("MODEL", False)
        config["kcor_file"] = self.options.get("KCOR_FILE", None)
        config["mode"] = mode
        config["job_base_name"] = self.job_base_name
        config["class_balanced"] = (mode == "train")

        types = self._get_types()
        if types is not None:
          types = {int(k): v for k, v in types.items()} # sometimes the keys are strings, sometimes ints
          self.logger.info(f"input types from sim found, types set to {types}")
          config["sn_type_id_to_name"] = types

        return yaml.dump(config)

    def _check_completion(self, squeue):
        if Path(self.done_file).exists():
            self.logger.debug(f"Found done file at {self.done_file}")
            with open(self.done_file) as f:
                if "SUCCESS" not in f.read().upper():
                    return Task.FINISHED_FAILURE

            pred_path = str(Path(self.output_dir) / "predictions.csv")
            predictions = pd.read_csv(pred_path)
            predictions = predictions[["snid", "pred_labels"]] # make sure snid is the first col
            predictions = predictions.rename(columns={"pred_labels": self.get_prob_column_name()})
            predictions.to_csv(pred_path, index=False)
            self.logger.info(f"Predictions file can be found at {pred_path}")
            self.output.update({"model_filename": self.options.get("MODEL", str(Path(self.output_dir) / "trained_model")), "predictions_filename": pred_path})
            return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_base_name)

    def _heatmap_creation_success(self):
        if not Path(self.heatmaps_done_file).exists():
            return False
        with open(self.heatmaps_done_file, "r") as donefile:
            if "CREATE HEATMAPS FAILURE" in donefile.read():
                return False
        return Path(self.heatmaps_path).exists() and (Path(self.heatmaps_path) / "done.log").exists()

    def num_jobs_in_queue(self):
        print("rerun num jobs in queue")
        squeue = [i.strip() for i in subprocess.check_output(f"squeue -h -u $USER -o '%.200j'", shell=True, text=True).splitlines()]
        self.logger.debug(f"{squeue}")
        return self.check_for_job(squeue, self.job_base_name)

    @staticmethod
    def _get_lcdata_paths(sim_dir):
        lcdata_paths = [str(f.resolve()) for f in Path(sim_dir).iterdir() if "PHOT" in f.name]
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
