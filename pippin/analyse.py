import inspect
import shutil
import subprocess
import os

from pippin.config import mkdirs, get_config
from pippin.task import Task


class AnalyseChains(Task):  # TODO: Define the location of the output so we can run the lc fitting on it.
    """ Smack the data into something that looks like the simulated data

    CONFIGURATION
    =============

    COSMOMC:
        label:
            MASK_COSMOMC: mask  # partial match
            OPTS:
                BLIND: [omegam, w]  # Optional. Parameters to blind. Single or list. omegam, omegal, w, wa, nu, etc
                COVOPTS: [ALL, NOSYS] # Optional. Covopts to match. Single or list. Exact match.

    OUTPUTS
    =======



    """
    def __init__(self, name, output_dir, options, dependencies=None):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.options = options
        self.global_config = get_config()

        self.logfile = os.path.join(self.output_dir, "output.log")
        self.job_name = f"anaylyse_chains_{name}"
        self.path_to_code = os.path.dirname(inspect.stack()[0][1]) + "/external/plot_cosmomc.py"

        self.covopts = options.get("COVOPTS")
        if isinstance(self.covopts, str):
            self.covopts = [self.covopts]

        self.files = []
        self.params = []
        self.blind_params = options.get("BLIND")

        # Assuming all deps are cosmomc tasks
        for c in self.dependencies:
            for covopt in c.output["covopts"]:
                if covopt in self.covopts or self.covopts is None:
                    self.files.append(c.output["base_dict"][covopt])
                    self.params += c.output["cosmology_params"]
        self.params = list(set(self.params))

        self.slurm = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=broadwl
#SBATCH --output={log_file}
#SBATCH --account=pi-rkessler
#SBATCH --mem=10GB

cd {output_dir}
python {path_to_code} {files} {name} {blind} {done_file} {params}
"""

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at f{self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read():
                    self.logger.error(f"Done file reported failure. Check output log {self.logfile}")
                    return Task.FINISHED_FAILURE
                else:
                    return Task.FINISHED_SUCCESS
        return 1

    def _run(self, force_refresh):

        format_dict = {
            "job_name": self.job_name,
            "log_file": self.logfile,
            "done_file": "-d " + self.done_file,
            "path_to_code": self.path_to_code,
            "output_dir": self.output_dir,
            "files": " ".join(self.files),
            "name": "-n " + self.name,
            "blind": ("-b " + " ".join(self.blind_params)) if self.blind_params else "",
            "params": "-p " + " ".join(self.params)
        }
        final_slurm = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string(final_slurm)
        old_hash = self.get_old_hash()

        if force_refresh or new_hash != old_hash:
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)
            slurm_output_file = os.path.join(self.output_dir, "slurm.job")
            with open(slurm_output_file, "w") as f:
                f.write(final_slurm)
            self.logger.info(f"Submitting batch job for data prep")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")
        return True
