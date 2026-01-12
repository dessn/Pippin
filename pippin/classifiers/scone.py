# Created Mar 2024 by R.Kessler and H.Qu
# Refactor pippin interface to scone to accept and modify
# a scone-input file.

import os
import shutil
import subprocess
from pathlib import Path

from pippin.classifiers.classifier import Classifier
from pippin.config import get_config, get_data_loc
from pippin.task import Task


# =========================================

SCONE_SHELL_SCRIPT = "run.py"  # top-level script under $SCONE_DIR

KEYLIST_SCONE_INPUT = [
    "init_env_train",
    "init_env_heatmaps",
    "prescale_heatmaps",
    "nevt_select_heatmaps",
    "batch_size",
    "categorical",
    "class_balanced",
    "num_epochs",
    "num_mjd_bins",
    "num_wavelength_bins",
    "mode",
    "trained_model",
    "prob_column_name",
]


# KEY_PROB_COLUMN_NAME = "PROB_COLUMN_NAME"  # RK - Jan 2026


# ==========================================
class SconeClassifier(Classifier):
    """convolutional neural network-based SN photometric classifier
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
          SIM_FRACTION:  1 # fraction of sims to use for training (to be obsolete)
          PRESCALE_HEATMAPS:  1 # divide sample by PRESCALE for heatmag and training
          SCONE_CPU_BATCH_FILE: /path/to/sbatch/template/for/scone
          SCONE_GPU_BATCH_FILE: /path/to/sbatch/template/for/scone
          BATCH_REPLACE: {}

    OUTPUTS:
    ========
      predictions.csv: list of snids and associated predictions
      training_history.csv: training history output from keras

    """

    def __new__(
        cls,
        name,
        output_dir,
        config,
        dependencies,
        mode,
        options,
        index=0,
        model_name=None,
    ):
        # XXX DEPRECATION
        # If no BASE file is present, run legacy version of Scone
        # Avoid recursive nonsense by making sure the type of `cls` is SconeClassifier
        if cls == SconeClassifier and config.get("BASE") is None:
            # Have to import later because SconeClassifier must exist prior to importing SconeLegacyClassifier
            from pippin.classifiers.scone_legacy import SconeLegacyClassifier

            cls = SconeLegacyClassifier
        return super().__new__(cls)

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
        self.options = options

        # - - - - - - -
        # special checks to help users cope with some changes
        if mode == "predict" and "MODEL" in options:
            self.options["TRAINED_MODEL"] = self.options["MODEL"]

        self.gpu = self.options.get("GPU", False)
        self.init_env_heatmaps = self.global_config["SCONE"]["init_env_cpu"]
        self.init_env = (
            self.global_config["SCONE"]["init_env_cpu"]
            if not self.gpu
            else self.global_config["SCONE"]["init_env_gpu"]
        )
        self.path_to_classifier = self.global_config["SCONE"]["location"]

        self.combine_mask = "COMBINE_MASK" in config

        self.select_lcfit = self.options.get("OPTIONAL_MASK_FIT", None)  # RK May 3 2024
        scone_input_file = config.get(
            "BASE"
        )  # refactor by passing scone input file to pippin
        if scone_input_file is not None:
            scone_input_file = get_data_loc(scone_input_file)
        self.scone_input_file = scone_input_file

        output_path_obj = Path(self.output_dir)
        heatmaps_path_obj = output_path_obj / "heatmaps"

        self.job_base_name = (
            output_path_obj.parents[1].name + "__" + output_path_obj.name
        )

        self.batch_replace = self.options.get(
            "BATCH_REPLACE", self.global_config.get("BATCH_REPLACE", {})
        )

        self.heatmaps_done_file = str(heatmaps_path_obj / "done.txt")

        remake_heatmaps = self.options.get("REMAKE_HEATMAPS", False)
        self.keep_heatmaps = not remake_heatmaps

        return

    def classify(self, mode):
        self.logger.info(
            f"============ Prepare refactored SCONE with mode = {mode} ============="
        )
        failed = False
        if Path(self.done_file).exists():
            self.logger.debug(f"Found done file at {self.done_file}")
            with open(self.done_file) as f:
                if "SUCCESS" not in f.read().upper():
                    failed = True

        scone_input_file = self.scone_input_file

        # - - - -
        sim_deps = self.get_simulation_dependency()
        sim_dirs = [
            sim_dep.output["photometry_dirs"][self.index] for sim_dep in sim_deps
        ]

        # prepare scone input lines needed to create hash,
        # but don't create scone input file yet.
        scone_input_lines = self.prepare_scone_input_lines(sim_dirs, mode)

        str_config = " ".join(scone_input_lines)
        new_hash = self.get_hash_from_string(str_config)

        if self._check_regenerate(new_hash) or failed:
            self.logger.debug("Regenerating scone")
        else:
            self.logger.info("scone hash check passed, not rerunning")
            self.should_be_done()
            return True

        # later, perhaps check to preserve heatmaps ??
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        # write scone input file, and beware that name of scone
        # input file is updated
        scone_input_base = os.path.basename(self.scone_input_file)
        self.scone_input_file = self.output_dir + "/" + "PIP_" + scone_input_base
        with open(self.scone_input_file, "wt") as i:
            for line in scone_input_lines:
                i.write(f"{line}\n")

        self.save_new_hash(new_hash)

        path = Path(self.path_to_classifier) / SCONE_SHELL_SCRIPT
        path = (
            path
            if path.exists()
            else Path(self.path_to_classifier) / SCONE_SHELL_SCRIPT
        )
        cmd = f"python {str(path)} --config_path {self.scone_input_file} "
        #      f"--sbatch_job_name {self.job_base_name} "

        self.logger.info(f"Running command: {cmd}")
        subprocess.run([cmd], shell=True)

        return True

    def prepare_scone_input_lines(self, sim_dirs, mode):
        # Created Apr 2024 by R.Kessler
        # Read base scone input and make a few modification such as
        # the sim data dirs, and other substitutions defined in pippin input.
        # Method returns list of lines for modified scone-config input file.
        # Original comments and input layout are preserved.

        config_lines = []
        scone_input_file = self.scone_input_file
        options_local = self.options.copy()  # make local copy

        # set local mode as if it were an override key in pippin input file
        options_local["MODE"] = mode

        # fetch name of output column with scone prob_Ia
        if mode == "predict":
            options_local["PROB_COLUMN_NAME"] = self.get_prob_column_name()

        # - - - -
        flag_remove_line = False

        with open(scone_input_file, "r") as i:
            inp_config = i.read().split("\n")

        key_replace_dict = {}
        key_remove_list = [
            "input_data_paths:",
            "snid_select_files:",
            "sbatch_job_name:",
        ]

        for line_in in inp_config:
            line_out = line_in
            wdlist = line_in.split()
            nwd = len(wdlist)
            if nwd == 0:
                flag_remove_line = False
            else:
                if wdlist[0] == "output_path:":
                    line_out = line_in.replace(wdlist[1], self.output_dir)

                # goofy logic to remove original input_data_paths
                if flag_remove_line and wdlist[0] != "-":
                    flag_remove_line = False
                if wdlist[0] in key_remove_list:
                    flag_remove_line = True

                # check all possible scone keys that can be overwritten/added
                for key in KEYLIST_SCONE_INPUT:
                    if wdlist[0] == key + ":":
                        key_pippin = key.upper()
                        if key_pippin in options_local:
                            key_replace_dict[key_pippin] = True
                            val_replace = options_local[key_pippin]
                            line_out = line_in.replace(wdlist[1], str(val_replace))

                # remove prescale for predict mode
                if mode == "predict" and "prescale" in wdlist[0]:
                    line_out = f"# WARNING: {wdlist[0]} removed for {mode} mode."

            if not flag_remove_line:
                config_lines.append(line_out)

        # - - - - - - - - - -
        # add extra info from pippin
        config_lines.append("")
        config_lines.append("# ======================================= ")
        config_lines.append("# keys added by pippin\n ")

        # pass sbatch_job_name via config since there are other sbatch config
        # keys already. Could also pass via command line arg --sbatch_job_name.
        config_lines.append(f"sbatch_job_name: {self.job_base_name}\n")

        config_lines.append("input_data_paths:")
        for sim_dir in sim_dirs:
            resolved_dir = os.path.realpath(sim_dir)
            config_lines.append(f"  - {resolved_dir}")

        # add pippin-specified keys that were not in the original scone input
        for key_pippin in options_local:
            key = key_pippin.lower()
            if key_pippin not in key_replace_dict and key in KEYLIST_SCONE_INPUT:
                val = options_local[key_pippin]
                line = f"{key}:  {val}"
                config_lines.append("")
                config_lines.append(f"{line}")

        # check option to select events passing LCFIT

        if self.select_lcfit:
            config_lines.append("")
            config_lines.append("# Train on events passing LCFIT")
            config_lines.append("snid_select_files:")
            lcfit_deps = self.get_fit_dependency()
            # self.logger.info(f"\n xxx lcfit_deps = \n{lcfit_deps}\n")
            for tmp_dict in lcfit_deps:
                fitres_dir = tmp_dict["fitres_dirs"][self.index]
                fitopt_base_file = tmp_dict["fitopt_map"]["DEFAULT"]
                fitres_file = f"{fitres_dir}/{fitopt_base_file}"
                config_lines.append(f"  - {fitres_file}")

        return config_lines

    # def get_optional_requirements(config):
    #    # Created May 3 2024 by R.Kessler and P.Armstrong
    #    if config.get("SELECT_LCFIT", False):
    #        return False, True       # wait for LCFIT task
    #    return False, False          # no optional LCFIT task

    def predict(self):
        return self.classify("predict")

    def train(self):
        return self.classify("train")

    def _check_completion(self, squeue):
        if Path(self.done_file).exists():
            self.logger.debug(f"Found scone done file at {self.done_file}")
            with open(self.done_file) as f:
                if "SUCCESS" not in f.read().upper():
                    return Task.FINISHED_FAILURE

            pred_path = str(Path(self.output_dir) / "predictions.csv")

            self.output.update(
                {
                    "model_filename": self.options.get(
                        "MODEL", str(Path(self.output_dir) / "trained_model")
                    ),
                    "predictions_filename": pred_path,
                }
            )

            return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_base_name)

    def _heatmap_creation_success(self):
        if not Path(self.heatmaps_done_file).exists():
            return False
        with open(self.heatmaps_done_file, "r") as donefile:
            if "CREATE HEATMAPS FAILURE" in donefile.read():
                return False
        return (
            Path(self.heatmaps_path).exists()
            and (Path(self.heatmaps_path) / "done.log").exists()
        )

    @staticmethod
    def get_requirements(options):
        return True, False
