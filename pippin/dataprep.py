import shutil
import subprocess
import os

from pippin.config import mkdirs, get_output_loc, get_config
from pippin.task import Task


class DataPrep(Task):  # TODO: Define the location of the output so we can run the lc fitting on it.
    """ Smack the data into something that looks like the simulated data


    """
    def __init__(self, name, output_dir, options, dependencies=None):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.options = options
        self.global_config = get_config()

        self.logfile = os.path.join(self.output_dir, "output.log")
        self.conda_env = self.global_config["DataSkimmer"]["conda_env"]
        self.path_to_task = output_dir

        self.raw_dir = self.options.get("RAW_DIR")
        self.genversion = os.path.basename(self.raw_dir)
        self.data_path = os.path.dirname(self.raw_dir)
        self.job_name = f"DATAPREP_{self.name}"

        self.output["genversion"] = self.genversion
        self.output["photometry_dir"] = get_output_loc(self.raw_dir)
        self.output["raw_dir"] = self.raw_dir
        self.clump_file = os.path.join(self.output_dir, self.genversion + ".SNANA.TEXT")
        self.output["clump_file"] = self.clump_file

        self.slurm = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=broadwl
#SBATCH --output={log_file}
#SBATCH --account=pi-rkessler
#SBATCH --mem=2GB

cd {path_to_task}
snana.exe clump.nml
if [ $? -eq 0 ]; then
    echo SUCCESS > {done_file}
else
    echo FAILURE > {done_file}
fi
"""
        self.clump_command = """#
# Obtaining Clump fit
# to run:
# snana.exe SNFIT_clump.nml
# outputs csv file with space delimiters

  &SNLCINP

     ! For SNN-integration:
     OPT_SETPKMJD = 16
     SNTABLE_LIST = 'SNANA(text:key)'
     TEXTFILE_PREFIX = '{genversion}'

     ! data
     PRIVATE_DATA_PATH = '{data_path}'
     VERSION_PHOTOMETRY = '{genversion}'

     PHOTFLAG_MSKREJ   = 1016 !PHOTFLAG eliminate epoch that has errors, not LC 

  &END
"""
    def _get_types(self):
        self.logger.warning("Data does not report types, let's hope the defaults are up to date!")
        return None

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at f{self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read():
                    self.logger.info(f"Done file reported failure. Check output log {self.logfile}")
                    return Task.FINISHED_FAILURE
                else:
                    self.output["types"] = self._get_types()
                    return Task.FINISHED_SUCCESS
        return 1  # The number of CPUs being utilised

    def _run(self, force_refresh):

        command_string = self.clump_command.format(genversion=self.genversion, data_path=self.data_path)

        format_dict = {
            "job_name": self.job_name,
            "log_file": self.logfile,
            "path_to_task": self.path_to_task,
            "done_file": self.done_file
        }
        final_slurm = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string(command_string + final_slurm)
        old_hash = self.get_old_hash()

        if force_refresh or new_hash != old_hash:
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
            self.logger.info("Hash check passed, not rerunning")
        return True
