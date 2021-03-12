import os
import shutil
import subprocess
from pathlib import Path
import yaml

from pippin.classifiers.classifier import Classifier
from pippin.config import get_config, get_output_loc, mkdirs
from pippin.task import Task
from pippin.external.SNANA_FITS_to_pd import read_fits


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
          FITOPT: someLabel # Exact match to fitopt in a fitopt file. USED FOR TRAINING ONLY
          FEATURES: x1 c zHD  # Columns out of fitres file to use as features
          MODEL: someName # exact name of training classification task

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
        self.options = options

        self.conda_env = self.global_config["SCONE"]["conda_env"]
        self.path_to_classifier = self.global_config["SCONE"]["location"]

        self.job_base_name = os.path.basename(Path(output_dir).parents[1]) + "__" + os.path.basename(output_dir)

        self.config_path = os.path.join(self.output_dir, "model_config.yml")
        self.heatmaps_path = os.path.join(self.output_dir, "heatmaps")
        self.csvs_path = os.path.join(self.output_dir, "sim_csvs")
        self.slurm = """{sbatch_header}
        {task_setup}"""

        self.logfile = os.path.join(self.output_dir, "output.log")

        remake_heatmaps = self.options.get("REMAKE_HEATMAPS")
        self.keep_heatmaps = not remake_heatmaps if remake_heatmaps is not None else True

    def classify(self, mode):
        if self.gpu:
            self.sbatch_header = self.sbatch_gpu_header
        else:
            self.sbatch_header = self.sbatch_cpu_header

        header_dict = {
                "job-name": self.job_base_name,
                "output": "output.log",
                "time": "00:55:00", # TODO: scale based on number of heatmaps
                "mem-per-cpu": "8GB",
                "ntasks": "1",
                "cpus-per-task": "4"
                }
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
            if not os.path.exists(self.csvs_path):
                os.makedirs(self.csvs_path)
            metadata_paths, lcdata_paths = self._fitres_to_csv(self._get_lcdata_paths(sim_dirs), self.csvs_path)

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

    def _write_config_file(self, metadata_paths, lcdata_paths, mode, config_path):
        config = {}
        config["categorical"] = self.options.get("CATEGORICAL", False)
        config["num_epochs"] = self.options.get("NUM_EPOCHS")
        config["metadata_paths"] = metadata_paths
        config["lcdata_paths"] = lcdata_paths
        config["heatmaps_path"] = self.heatmaps_path
        config["num_wavelength_bins"] = self.options.get("NUM_WAVELENGTH_BINS", 32)
        config["num_mjd_bins"] = self.options.get("NUM_MJD_BINS", 180)
        config["Ia_fraction"] = self.options.get("IA_FRACTION", 0.5)
        config["donefile"] = self.done_file
        config["mode"] = mode
        config["sn_type_id_to_name"] = {42: "SNII",
          52: "SNIax",
          62: "SNIbc",
          67: "SNIa-91bg",
          64: "KN",
          90: "SNIa",
          95: "SLSN-1"}

        with open(config_path, "w+") as cfgfile:
            cfgfile.write(yaml.dump(config))

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Found done file at {self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read().upper():
                    return Task.FINISHED_FAILURE
                return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_base_name)

    def _heatmap_creation_success(self):
        return os.path.exists(self.heatmaps_path) and os.path.exists(os.path.join(self.heatmaps_path, "done.log")) 
        
    @staticmethod
    def _get_lcdata_paths(sim_dirs):
        sim_paths = [f.path for sim_dir in sim_dirs for f in os.scandir(sim_dir)]
        lcdata_paths = [path for path in sim_paths if "PHOT" in path]

        return lcdata_paths

    @staticmethod
    def _fitres_to_csv(lcdata_paths, output_dir):
        csv_metadata_paths = []
        csv_lcdata_paths = []

        for path in lcdata_paths:
            csv_metadata_path = os.path.join(output_dir, os.path.basename(path).replace("PHOT.FITS.gz", "HEAD.csv"))
            csv_lcdata_path = os.path.join(output_dir, os.path.basename(path).replace(".FITS.gz", ".csv"))

            if os.path.exists(csv_metadata_path) and os.path.exists(csv_lcdata_path):
                csv_metadata_paths.append(csv_metadata_path)
                csv_lcdata_paths.append(csv_lcdata_path)
                continue

            metadata, lcdata = read_fits(path)

            metadata.to_csv(csv_metadata_path)
            lcdata.to_csv(csv_lcdata_path)

            csv_metadata_paths.append(csv_metadata_path)
            csv_lcdata_paths.append(csv_lcdata_path)

        return csv_metadata_paths, csv_lcdata_paths

    @staticmethod
    def get_requirements(options):
        return True, False
