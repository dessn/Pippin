import os
from pathlib import Path
import yaml
import shutil
import subprocess

from pippin.base import ConfigBasedExecutable
from pippin.task import Task
from pippin.biascor import BiasCor
import pippin.cosmofitters.cosmomc as cosmomc
from pippin.config import get_data_loc, get_config, read_yaml, mkdirs, chown_dir

class SBCreateCov(ConfigBasedExecutable):
    """ Create covariance matrices and data from salt2mu used for cosmomc and wfit.
    Run through submit_batch

    CONFIGURATION:
    ==============
    CREATE_COV:
        label:
            OPTS:
              SUBTRACT_VPEC: False # Subtract VPEC contribution from MUERR if True
              SYS_SCALE: location of the fitopts file with scales in it
              FITOPT_SCALES:  # Optional dict to scale fitopts
                    fitopt_label_for_partial check: float to scale by  # (does label in fitopt, not exact match
              MUOPT_SCALES: # Optional dict used to construct SYSFILE input by putting MUOPT scales at the bottom, scale defaults to one
                exact_muopt_name: float
              COVOPTS:  # optional, note you'll get an 'ALL' covopt no matter what
                - "[NOSYS] [=DEFAULT,=DEFAULT]"  # syntax for Dan&Dillons script. [label] [fitopts_to_match,muopts_to_match]. Does partial matching. =Default means dont do that systematic type
              BATCH_INFO: sbatch $SBATCH_TEMPLATES/blah.TEMPLATE 1
              JOB_MAX_WALLTIME: 00:10:00
    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        ini_dir : The directory the .ini files for cosmomc will be output to
        covopts : a dictionary mapping a covopt label to a number
        blind: bool - whether or not to blind cosmo results

    """


    def __init__(self, name, output_dir, config, options, global_config, dependencies=None):
        
        base_file = get_data_loc("create_cov/COVMAT.input")
        super().__init__(name, output_dir, config, base_file, default_assignment=": ", dependencies = dependencies)

        if options is None:
            options = {}
        self.options = options
        self.templates_dir = self.options.get("INI_DIR", "cosmomc_templates")
        self.global_config = get_config()
        self.job_name = os.path.basename(Path(output_dir).parents[1]) + "_CREATE_COV_" + name
        self.config_dir = os.path.join(self.output_dir, "output")
        self.wfit_inpdir = []
        for d in dependencies:
            num_dirs = len(d.output["subdirs"])
            if num_dirs > 1:
                for i in range(num_dirs):
                    self.wfit_inpdir.append(os.path.join(self.config_dir, f"{self.name}_{d.name}_OUTPUT_BBCFIT-{str(i+1).zfill(4)}"))
            else:
                self.wfit_inpdir.append(os.path.join(self.config_dir, f"{self.name}_{d.name}_OUTPUT_BBCFIT"))
        self.done_file = os.path.join(self.config_dir, "ALL.DONE")
        self.input_name = f"{self.job_name}.INPUT"
        self.input_file = os.path.join(self.output_dir, self.input_name)
        self.logfile = os.path.join(self.output_dir, "output.log")

        # BATCH Options
        BATCH_INFO = self.options.get("BATCH_INFO")
        if BATCH_INFO is None:
            BATCH_FILE = self.options.get("BATCH_FILE")
            if BATCH_FILE is not None:
                BATCH_FILE = get_data_loc(BATCH_FILE)
            else:
                if self.gpu:
                    BATCH_FILE = self.global_config["SBATCH"]["gpu_location"]
                else:
                    BATCH_FILE = self.global_config["SBATCH"]["cpu_location"]
            num_jobs = 0
            for d in dependencies:
                num_jobs += len(d.output["subdirs"])
            # To make sure we never ask for too many cores
            if num_jobs > 20:
                num_jobs = 20
            BATCH_INFO = f"sbatch {BATCH_FILE} {num_jobs}"
        BATCH_REPLACE = self.options.get("BATCH_REPLACE", {})
        if BATCH_REPLACE != {}:
            BATCH_MEM = BATCH_REPLACE.get("REPLACE_MEM", "2GB")
            BATCH_WALLTIME = BATCH_REPLACE.get("REPLACE_WALLTIME", "24:00:00")
        else:
            BATCH_MEM = self.options.get("BATCH_MEM", "2GB")
            BATCH_WALLTIME = self.options.get("BATCH_WALLTIME", "24:00:00")
        self.yaml["CONFIG"]["BATCH_INFO"] = BATCH_INFO
        self.yaml["CONFIG"]["BATCH_MEM"] = BATCH_MEM
        self.yaml["CONFIG"]["BATCH_WALLTIME"] = BATCH_WALLTIME

        # create_covariance.py input file
        self.input_covmat_file = get_data_loc("create_cov/input_file.txt") 
        self.output_covmat_file = os.path.join(self.output_dir, "input_file.txt")
        self.prepare_cosmomc = self.config.get("COSMOMC", False)
        if self.prepare_cosmomc:
            self.logger.info("Generating CosmoMC output")
        else:
            self.logger.info("Not generating CosmoMC output") 
        self.sys_file_in = self.get_sys_file_in()
        self.sys_file_out = os.path.join(self.output_dir, "sys_scale.yml")
        self.calibration_set = options.get("CALIBRATORS", [])
        self.prepare_input_covmat()
        self.yaml["CONFIG"]["INPUT_COVMAT_FILE"] = self.output_covmat_file

        # Rest of the submit_batch input file
        self.bbc_outdirs = self.get_bbc_outdirs()
        self.yaml["CONFIG"]["BBC_OUTDIR"] = self.bbc_outdirs

        self.covmatopt = self.get_covmatopt()
        self.yaml["CONFIG"]["COVMATOPT"] = [self.covmatopt]

        self.yaml["CONFIG"]["OUTDIR"] = self.config_dir

        # Output
        self.output["blind"] = [d.output["blind"] for d in self.dependencies]
        self.output["hubble_plot"] = [d.output["hubble_plot"] for d in self.dependencies]
        covopts_map = {"ALL": 0}
        for i, covopt in enumerate(self.options.get("COVOPTS", [])):
            covopts_map[covopt.split("]")[0][1:]] = i + 1
        self.output["covopts"] = covopts_map
        self.output["ini_dir"] = os.path.join(self.config_dir, "cosmomc")
        self.output["index"] = 0
        self.output["bcor_name"] = [d.name for d in self.dependencies]

    def add_dependent(self, task):
        self.dependents.append(task)
        if isinstance(task, cosmomc.CosmoMC):
            self.logger.info("CosmoMC task found, CreateCov will generate CosmoMC output")
            self.prepare_cosmomc = True

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at {self.done_file}")
            with open(self.done_file) as f:
                if "FAIL" in f.read():
                    self.logger.error(f"Done file reported failure. Check output log {self.logfile}")
                    self.scan_files_for_error([self.logfile], "ERROR", "EXCEPTION")
                    return Task.FINISHED_FAILURE
                else:
                    return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_name)

    def load_input_covmat(self):
        with open(self.input_covmat_file, "r") as f:
            file_lines = list(f.read().splitlines())
        for index, line in enumerate(file_lines):
            if "#END_YAML" in line:
                yaml_str = "\n".join(file_lines[:index])
                self.input_covmat_yaml = yaml.safe_load(yaml_str)
                break

    def prepare_input_covmat(self):
        self.load_input_covmat()
        if self.prepare_cosmomc:
            self.input_covmat_yaml["COSMOMC_TEMPLATES_PATH"] = get_data_loc(self.templates_dir)
        else:
            self.input_covmat_yaml.pop("COSMOMC_TEMPLATES_PATH", None)
        self.input_covmat_yaml["SYS_SCALE_FILE"] = self.sys_file_out
        self.input_covmat_yaml["COVOPTS"] = self.options.get("COVOPTS", [])
        self.input_covmat_yaml["EXTRA_COVS"] = self.options.get("EXTRA_COVS", [])
        self.input_covmat_yaml["CALIBRATORS"] = self.calibration_set

    def get_bbc_outdirs(self):
        bbc_outdirs = []
        for d in self.dependencies:
            name = d.name
            output = d.fit_output_dir
            bbc_outdirs.append(f"/{name}/    {output}")
        return bbc_outdirs

    def get_covmatopt(self):
        rebin_x1 = self.options.get("REBINNED_X1", 0)
        rebin_c = self.options.get("REBINNED_C", 0)
        if (rebin_x1 + rebin_c > 0):
            if (rebin_x1 == 0) or (rebin_c == 0):
                Task.fail_config(f"If rebin, both REBINNED_X1 ({rebin_x1}) and REBINNED_C ({rebin_c}) must be greater than 0")
            else:
                cmd = f"--nbin_x1 {rebin_x1} --nbin_c {rebin_c}"
        elif self.options.get("SUBTRACT_VPEC", False):
            cmd = "--subtract_vpec"
        elif self.options.get("BINNED", True):
            cmd = ""
        else:
            cmd = "--unbinned"
        return f"/{self.name}/    {cmd}"

    def get_sys_file_in(self):
        set_file = self.options.get("SYS_SCALE")
        if set_file is not None:
            self.logger.debug(f"Explicit SYS_SCALE file specified: {set_file}")
            path = get_data_loc(set_file)
            if path is None:
                raise ValueError(f"Unable to resolve path to {set_file}")
        else:
            self.logger.debug("Searching for SYS_SCALE source from biascor task")
            path = None
            for d in self.dependencies:
                fitopt_files = []
                fitopt_files += [f for f in d.output["fitopt_files"] if f is not None]
                assert len(set(fitopt_files)) < 2, f"Cannot automatically determine scaling from FITOPT file as you have multiple files: {fitopt_files}"
                if (len(fitopt_files) > 0) and (path is None):
                    path = fitopt_files[0]
                    break
        self.options["SYS_SCALE"] = path  # Save to options so its serialised out
        self.logger.info(f"Setting systematics scaling file to {path}")
        return path

    def get_scales_from_fitopt_file(self):
        if self.sys_file_in is None:
            return {}
        self.logger.debug(f"Loading sys scaling from {self.sys_file_in}")
        yaml = read_yaml(self.sys_file_in)
        if 'FLAG_USE_SAME_EVENTS' in yaml.keys():
            yaml.pop('FLAG_USE_SAME_EVENTS')
        raw = {k: float(v.split(maxsplit=1)[0]) for _, d in yaml.items() for k, v in d.items()}
        return raw

    def get_sys_scale(self):
        return {**self.get_scales_from_fitopt_file(), **self.options.get("FITOPT_SCALES", {})}

    def _run(self):
        sys_scale = self.get_sys_scale()
        self.logger.debug(f"Final sys_scale: {sys_scale}")

        final_output_for_hash = self.get_output_string()

        new_hash = self.get_hash_from_string(final_output_for_hash)

        if self._check_regenerate(new_hash):
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)

            with open(self.input_file, "w") as f:
                f.write(self.get_output_string())

            with open(self.sys_file_out, "w") as f:
                f.write(yaml.safe_dump(sys_scale, width=2048))

            with open(self.output_covmat_file, "w") as f:
                f.write(yaml.safe_dump(self.input_covmat_yaml, width=2048))

            cmd = ["submit_batch_jobs.sh", os.path.basename(self.input_file)]
            self.logger.debug(f"Submitting CreateCov job: {' '.join(cmd)} in cwd: {self.output_dir}")
            self.logger.debug(f"Logging to {self.logfile}")
            with open(self.logfile, 'w') as f:
                subprocess.run(' '.join(cmd), stdout=f, stderr=subprocess.STDOUT, cwd=self.output_dir, shell=True)
            chown_dir(self.output_dir)
        else:
            self.should_be_done()
            self.logger.info("Hash check passed, not rerunning")
        return True

        
    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):

        biascor_tasks = Task.get_task_of_type(prior_tasks, BiasCor)
        
        def _get_createcov_dir(base_output_dir, stage_number, name):
            return f"{base_output_dir}/{stage_number}_CREATE_COV/{name}"

        tasks = []
        for cname in c.get("CREATE_COV", []):
            config = c["CREATE_COV"][cname]
            if config is None:
                config = {}
            options = config.get("OPTS", {})
            mask = config.get("MASK", config.get("MASK_BIASCOR", ""))

            btasks = [btask for btask in biascor_tasks if mask in btask.name]
            if len(btasks) == 0:
                Task.fail_config(f"Create cov task {cname} has no biascor tasks matching mask {mask}")

            t = SBCreateCov(cname, _get_createcov_dir(base_output_dir, stage_number, cname), config, options, global_config, dependencies=btasks)
            tasks.append(t)

        return tasks
