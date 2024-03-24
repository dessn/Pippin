import shutil
import subprocess
import tarfile
import os
from collections import OrderedDict
from pathlib import Path
from pippin.config import mkdirs, get_output_loc, get_config, get_data_loc, read_yaml, merge_dict
from pippin.task import Task


class DataPrep(Task):  # TODO: Define the location of the output so we can run the lc fitting on it.
    """ Smack the data into something that looks like the simulated data

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        genversion : Genversion
        data_path : dir with all data in it (dir above raw_dir)
        photometry_dir : dir with fits file photometry in it
        raw_dir: input directory
        clump_file: clumping file with estimate of t0 for each event
        types_dict: dict mapping IA and NONIA to types
        types: dict mapping numbers to types, used by Supernnova
        blind: bool - whether or not to blind cosmo results
        is_sim: bool - whether or not the input is a simulation
    """

    def __init__(self, name, output_dir, config, options, global_config, dependencies=None):
        super().__init__(name, output_dir, config=config, dependencies=dependencies)
        self.options = options
        self.global_config = get_config()

        self.logfile = os.path.join(self.output_dir, "output.log")
        self.conda_env = self.global_config["DataSkimmer"]["conda_env"]
        self.path_to_task = output_dir

        self.unparsed_raw = self.options.get("RAW_DIR")
        self.raw_dir = get_data_loc(self.unparsed_raw)
        if self.raw_dir is None:
            Task.fail_config(f"Unable to find {self.options.get('RAW_DIR')}")

        if self.raw_dir[-1] == "/":
            self.raw_dir = self.raw_dir[:-1]
        self.genversion = os.path.basename(self.raw_dir)
        self.data_path = os.path.dirname(self.raw_dir)
        if self.unparsed_raw == "$SCRATCH_SIMDIR" or "SNDATA_ROOT/SIM" in self.raw_dir:
            self.logger.debug("Removing PRIVATE_DATA_PATH from NML file")
            self.data_path = ""
        self.job_name = os.path.basename(Path(output_dir).parents[1]) + "_DATAPREP_" + self.name

        self.output_info = os.path.join(self.output_dir, f"{self.genversion}.YAML")
        self.output["genversion"] = self.genversion
        self.opt_setpkmjd = options.get("OPT_SETPKMJD", 16)
        self.photflag_mskrej = options.get("PHOTFLAG_MSKREJ", 1016)
        self.output["data_path"] = self.data_path
        self.output["photometry_dirs"] = [get_output_loc(self.raw_dir)]
        self.output["sim_folders"] = [get_output_loc(self.raw_dir)]
        self.output["raw_dir"] = self.raw_dir
        self.clump_file = os.path.join(self.output_dir, self.genversion + ".SNANA.TEXT")
        self.output["clump_file"] = self.clump_file
        self.output["ranseed_change"] = False
        is_sim = options.get("SIM", False)
        self.output["is_sim"] = is_sim
        self.output["blind"] = options.get("BLIND", True)

        self.types_dict = options.get("TYPES")
        if self.types_dict is None:
            self.types_dict = {"IA": [1], "NONIA": [2, 20, 21, 22, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 80, 81]}
        else:
            for key in self.types_dict.keys():
                self.types_dict[key] = [int(c) for c in self.types_dict[key]]

        self.batch_file = self.options.get("BATCH_FILE")
        if self.batch_file is not None:
            self.batch_file = get_data_loc(self.batch_file)
        self.batch_replace = self.options.get("BATCH_REPLACE", self.global_config.get("BATCH_REPLACE", {}))

        self.logger.debug(f"\tIA types are {self.types_dict['IA']}")
        self.logger.debug(f"\tNONIA types are {self.types_dict['NONIA']}")
        self.output["types_dict"] = self.types_dict
        self.types = OrderedDict()
        for n in self.types_dict["IA"]:
            self.types.update({n: "Ia"})
        for n in self.types_dict["NONIA"]:
            self.types.update({n: "II"})
        self.output["types"] = self.types

        self.slurm = """{sbatch_header}
        {task_setup}"""

        self.clump_command = """#
# Obtaining Clump fit
# to run:
# snana.exe SNFIT_clump.nml
# outputs csv file with space delimiters

  &SNLCINP

     ! For SNN-integration:
     OPT_SETPKMJD = {opt_setpkmjd}
     SNTABLE_LIST = 'SNANA(text:key)'
     TEXTFILE_PREFIX = '{genversion}'
     OPT_YAML = 1

     ! data
     PRIVATE_DATA_PATH = '{data_path}'
     VERSION_PHOTOMETRY = '{genversion}'

     PHOTFLAG_MSKREJ   = {photflag_mskrej} !PHOTFLAG eliminate epoch that has errors, not LC 

     {photflag}
     {cutwin_snr_nodetect}

  &END
"""

    def _get_types(self):
        return self.types

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at {self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read():
                    self.logger.info(f"Done file reported failure. Check output log {self.logfile}")
                    return Task.FINISHED_FAILURE
                else:
                    if not os.path.exists(self.output_info):
                        self.logger.exception(f"Cannot find output info file {self.output_info}")
                        return Task.FINISHED_FAILURE
                    else:
                        content = read_yaml(self.output_info)
                        self.output["SURVEY"] = content["SURVEY"]
                        self.output["SURVEY_ID"] = content["IDSURVEY"]
                    self.output["types"] = self._get_types()
                    return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_name)

    def _run(self):

        val_p = self.options.get("PHOTFLAG_DETECT")
        val_c = self.options.get("CUTWIN_SNR_NODETECT")
        photflag = f"PHOTFLAG_DETECT = {val_p}" if val_p else ""
        cutwin = f"CUTWIN_SNR_NODETECT = {val_c}" if val_c else ""
        command_string = self.clump_command.format(
            genversion=self.genversion, data_path=self.data_path, opt_setpkmjd=self.opt_setpkmjd, photflag=photflag, cutwin_snr_nodetect=cutwin, photflag_mskrej=self.photflag_mskrej
        )
        
        if self.batch_file is None:
            if self.gpu:
                self.sbatch_header = self.sbatch_gpu_header
            else:
                self.sbatch_header = self.sbatch_cpu_header
        else:
            with open(self.batch_file, 'r') as f:
                self.sbatch_header = f.read()
            self.sbatch_header = self.clean_header(self.sbatch_header)

        header_dict = {
                "REPLACE_NAME": self.job_name,
                "REPLACE_WALLTIME": "0:20:00",
                "REPLACE_LOGFILE": self.logfile,
                "REPLACE_MEM": "2GB",
                "APPEND": ["#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1"]
                }
        header_dict = merge_dict(header_dict, self.batch_replace)
        self.update_header(header_dict)
        setup_dict = {
                "path_to_task": self.path_to_task,
                "done_file": self.done_file
                }
        format_dict = {
                "sbatch_header": self.sbatch_header,
                "task_setup": self.update_setup(setup_dict, self.task_setup['dataprep'])
                }
        #format_dict = {"job_name": self.job_name, "log_file": self.logfile, "path_to_task": self.path_to_task, "done_file": self.done_file}
        final_slurm = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string(command_string + final_slurm)

        if self._check_regenerate(new_hash):
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)
            slurm_output_file = os.path.join(self.output_dir, "slurm.job")
            clump_file = os.path.join(self.output_dir, "clump.nml")
            with open(slurm_output_file, "w") as f:
                f.write(final_slurm)
            with open(clump_file, "w") as f:
                f.write(command_string)

            self.logger.info(f"Submitting batch job for data prep")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)
        else:
            self.should_be_done()
            self.logger.info("Hash check passed, not rerunning")
        return True

    @staticmethod
    def get_tasks(config, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        tasks = []
        for name in config.get("DATAPREP", []):
            output_dir = f"{base_output_dir}/{stage_number}_DATAPREP/{name}"
            options = config["DATAPREP"][name].get("OPTS")
            if options is None and config["DATAPREP"][name].get("EXTERNAL") is None:
                Task.fail_config(f"DATAPREP task {name} needs to specify OPTS!")
            s = DataPrep(name, output_dir, config["DATAPREP"][name], options, global_config)
            Task.logger.debug(f"Creating data prep task {name} with {s.num_jobs} jobs, output to {output_dir}")
            tasks.append(s)
        return tasks
