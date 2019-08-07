import inspect
import shutil
import subprocess
import os

from pippin.base import ConfigBasedExecutable
from pippin.biascor import BiasCor
from pippin.config import mkdirs, get_config
from pippin.task import Task


class CreateCov(ConfigBasedExecutable):  # TODO: Define the location of the output so we can run the lc fitting on it.
    """ Smack the data into something that looks like the simulated data


    """
    def __init__(self, name, output_dir, options, dependencies=None):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/create_cov"
        self.template_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/cosmomc_templates"
        base_file = os.path.join(self.data_dir, "input_file.txt")
        super().__init__(name, output_dir, base_file, default_assignment=": ", dependencies=dependencies)

        self.options = options
        self.global_config = get_config()

        self.job_name = f"CREATE_COV_{name}"
        self.path_to_code = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/external")

        self.logfile = os.path.join(self.output_dir, "output.log")
        self.sys_file_in = os.path.join(self.data_dir, "sys_scale.LIST")
        self.sys_file_out = os.path.join(self.output_dir, "sys_scale.LIST")
        self.chain_dir = os.path.join(self.output_dir, "chains")
        self.config_dir = os.path.join(self.output_dir, "output")

        self.biascor_dep = self.get_dep(BiasCor, fail=True)
        self.input_file = os.path.join(self.output_dir, self.biascor_dep.output["subdir"] + ".input")


        self.slurm = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=broadwl-lc
#SBATCH --output={log_file}
#SBATCH --account=pi-rkessler
#SBATCH --mem=1GB

cd {path_to_code}
source activate
python create_covariance_staticbins.py {input_file} {done_file}
"""

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at f{self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read():
                    self.logger.info(f"Done file reported failure. Check output log {self.logfile}")
                    return Task.FINISHED_FAILURE
                else:
                    return Task.FINISHED_SUCCESS
        return 1  # The number of CPUs being utilised

    def calculate_input(self):
        mkdirs(self.config_dir)
        self.logger.debug(f"Calculating input")
        self.set_property("COSMOMC_TEMPLATES", self.template_dir)
        self.set_property("BASEOUTPUT", self.name)
        self.set_property("SYSFILE", self.sys_file_out)
        self.set_property("TOPDIR", self.biascor_dep.output["fit_output_dir"])
        self.set_property("OUTPUTDIR", self.config_dir)
        self.set_property("SUBDIR", self.biascor_dep.output["subdir"])
        self.set_property("ROOTDIR", self.chain_dir)
        self.set_property("SYSDEFAULT", self.options.get("SYSDEFAULT", 0))

        # More bs hacks
        covopt_str = ""
        for i, covopt in enumerate(self.options.get("COVOPTS", [])):
            if i > 0:
                covopt_str += "COVOPT: "
            covopt_str += covopt + "\n"
        self.set_property("COVOPT", covopt_str)

        # Load in sys file, add muopt arguments if needed
        # Get the MUOPT_SCALES and FITOPT scales keywords
        with open(self.sys_file_in) as f:
            sys_scale = f.read().splitlines()

            # Overwrite the fitopt scales
            fitopt_scale_overwrites = self.options.get("FITOPT_SCALES")
            for label, overwrite in fitopt_scale_overwrites.items():
                for i, line in enumerate(sys_scale):
                    comps = line.split()
                    if label in comps[1]:
                        sys_scale[i] = " ".join(comps[:-1] + [f"{overwrite}"])

            # Set the muopts scales
            muopt_scales = self.options.get("MUOPT_SCALES")
            muopts = self.biascor_dep.output["muopts"]
            for muopt in muopts:
                scale = muopt_scales.get(muopt, 1.0)
                sys_scale.append(f"ERRSCALE: DEFAULT {muopt} {scale}")

            return sys_scale

    def _run(self, force_refresh):
        sys_scale = self.calculate_input()
        format_dict = {
            "job_name": self.job_name,
            "log_file": self.logfile,
            "done_file": self.done_file,
            "path_to_code": self.path_to_code,
            "input_file": self.input_file
        }
        final_slurm = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string("\n".join(self.base + sys_scale) + final_slurm)
        old_hash = self.get_old_hash()

        if force_refresh or new_hash != old_hash:
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)
            # Write sys scales and the main input file
            with open(self.sys_file_out, "w") as f:
                f.write("\n".join(sys_scale))
            with open(self.input_file, "w") as f:
                f.write("\n".join(self.base))
            # Write out slurm job script
            slurm_output_file = os.path.join(self.output_dir, "slurm.job")
            with open(slurm_output_file, "w") as f:
                f.write(final_slurm)

            self.logger.info(f"Submitting batch job for data prep")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")
        return True
