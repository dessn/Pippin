import inspect
import shutil
import subprocess
import os
from pathlib import Path

from pippin.base import ConfigBasedExecutable
from pippin.biascor import BiasCor
from pippin.config import mkdirs, get_config, get_data_loc
from pippin.task import Task


class CreateCov(ConfigBasedExecutable):
    """ Create covariance matrices and data from salt2mu used for cosmomc

    CONFIGURATION:
    ==============
    CREATE_COV:
        label:
            OPTS:
              SYS_SCALE: location of sys_scale.LIST file (relative to create_cov dir by default)
              FITOPT_SCALES:  # Optional dict to scale fitopts
                    fitopt_label_for_partial check: float to scale by  # (does label in fitopt, not exact match
              MUOPT_SCALES: # Optional dict used to construct SYSFILE input by putting MUOPT scales at the bottom, scale defaults to one
                exact_muopt_name: float
              COVOPTS:  # optional, note you'll get an 'ALL' covopt no matter what
                - "[NOSYS] [=DEFAULT,=DEFAULT]"  # syntax for Dan&Dillons script. [label] [fitopts_to_match,muopts_to_match]. Does partial matching. =Default means dont do that systematic type

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        ini_dir : The directory the .ini files for cosmomc will be output to
        covopts : a dictionary mapping a covopt label to a number
        blind: bool - whether or not to blind cosmo results

    """

    def __init__(self, name, output_dir, config, options, global_config, dependencies=None, index=0):

        base_file = get_data_loc("create_cov/input_file.txt")
        super().__init__(name, output_dir, config, base_file, default_assignment=": ", dependencies=dependencies)

        if options is None:
            options = {}
        self.options = options
        self.templates_dir = self.options.get("INI_DIR", "cosmomc_templates")
        self.global_config = get_config()
        self.index = index
        self.job_name = os.path.basename(Path(output_dir).parents[1]) + "_CREATE_COV_" + name
        self.path_to_code = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/external")

        self.logfile = os.path.join(self.output_dir, "output.log")
        self.sys_file_in = get_data_loc(options.get("SYS_SCALE", "surveys/des/bbc/scale_5yr.yml"))
        self.sys_file_out = os.path.join(self.output_dir, "sys_scale.LIST")
        self.chain_dir = os.path.join(self.output_dir, "chains/")
        self.config_dir = os.path.join(self.output_dir, "output")

        self.biascor_dep = self.get_dep(BiasCor, fail=True)
        self.output["blind"] = self.biascor_dep.output["blind"]
        self.input_file = os.path.join(self.output_dir, self.biascor_dep.output["subdirs"][index] + ".input")
        self.output["hubble_plot"] = self.biascor_dep.output["hubble_plot"]

        self.output["ini_dir"] = self.config_dir
        covopts_map = {"ALL": 0}
        for i, covopt in enumerate(self.options.get("COVOPTS", [])):
            covopts_map[covopt.split("]")[0][1:]] = i + 1
        self.output["covopts"] = covopts_map
        self.output["index"] = index
        self.output["bcor_name"] = self.biascor_dep.name
        self.slurm = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=broadwl
#SBATCH --output={log_file}
#SBATCH --account=pi-rkessler
#SBATCH --mem=1GB

cd {output_dir}
conda activate
python {path_to_code}/create_covariance.py {input_file}
if [ $? -eq 0 ]; then
    echo SUCCESS > {done_file}
else
    echo FAILURE > {done_file}
fi
"""

    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at {self.done_file}")
            with open(self.done_file) as f:
                if "FAILURE" in f.read():
                    self.logger.error(f"Done file reported failure. Check output log {self.logfile}")
                    return Task.FINISHED_FAILURE
                else:
                    return Task.FINISHED_SUCCESS
        return self.check_for_job(squeue, self.job_name)

    def calculate_input(self):
        self.logger.debug(f"Calculating input")
        self.yaml["COSMOMC_TEMPLATES"] = get_data_loc(self.templates_dir)
        self.yaml["NAME"] = self.name
        self.yaml["SYSFILE"] = self.sys_file_out
        self.yaml["INPUT_DIR"] = self.biascor_dep.output["fit_output_dir"]
        self.yaml["OUTDIR"] = self.config_dir
        self.yaml["VERSION"] = self.biascor_dep.output["subdirs"][self.index]
        self.yaml["COVOPTS"] = self.options.get("COVOPTS", [])

        # Load in sys file, add muopt arguments if needed
        # Get the MUOPT_SCALES and FITOPT scales keywords
        self.logger.debug(f"Leading sys scaling from {self.sys_file_in}")
        with open(self.sys_file_in) as f:
            sys_scale = f.read().splitlines()

            # Overwrite the fitopt scales
            fitopt_scale_overwrites = self.options.get("FITOPT_SCALES", {})
            for label, overwrite in fitopt_scale_overwrites.items():
                for i, line in enumerate(sys_scale):
                    comps = line.split()
                    if label in comps[1]:
                        sys_scale[i] = " ".join(comps[:-1] + [f"{overwrite}"])
                        self.logger.debug(f"FITOPT_SCALES: Setting {' '.join(comps)} to {sys_scale[i]}")

            # Set the muopts scales
            muopt_scales = self.options.get("MUOPT_SCALES", {})
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
            "input_file": self.input_file,
            "output_dir": self.output_dir,
        }
        final_slurm = self.slurm.format(**format_dict)

        final_output_for_hash = self.get_output_string() + sys_scale + final_slurm

        new_hash = self.get_hash_from_string(final_output_for_hash)
        old_hash = self.get_old_hash()

        if force_refresh or new_hash != old_hash:
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            mkdirs(self.config_dir)
            self.save_new_hash(new_hash)
            # Write sys scales and the main input file
            with open(self.sys_file_out, "w") as f:
                f.write("\n".join(sys_scale))
            with open(self.input_file, "w") as f:
                f.write(self.get_output_string())
            # Write out slurm job script
            slurm_output_file = os.path.join(self.output_dir, "slurm.job")
            with open(slurm_output_file, "w") as f:
                f.write(final_slurm)

            self.logger.info(f"Submitting batch job for data prep")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir, shell=True)
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
            mask = config.get("MASK", "")

            for btask in biascor_tasks:
                if mask not in btask.name:
                    continue

                num = len(btask.output["subdirs"])
                for i in range(num):
                    ii = "" if num == 1 else f"_{i + 1}"

                    name = f"{cname}_{btask.name}{ii}"
                    a = CreateCov(name, _get_createcov_dir(base_output_dir, stage_number, name), config, options, global_config, dependencies=[btask], index=i)
                    Task.logger.info(f"Creating createcov task {name} for {btask.name} with {a.num_jobs} jobs")
                    tasks.append(a)

            if len(biascor_tasks) == 0:
                Task.fail_config(f"Create cov task {cname} has no biascor task to run on!")

        return tasks
