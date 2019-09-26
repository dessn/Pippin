import inspect
import shutil
import subprocess
import os
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

    """

    def __init__(self, name, output_dir, options, dependencies=None, index=0):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/create_cov"
        self.template_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/cosmomc_templates"
        base_file = os.path.join(self.data_dir, "input_file.txt")
        super().__init__(name, output_dir, base_file, default_assignment=": ", dependencies=dependencies)

        self.options = options
        self.global_config = get_config()
        self.index = index

        self.job_name = f"CREATE_COV_{name}"
        self.path_to_code = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/external")

        self.logfile = os.path.join(self.output_dir, "output.log")
        self.sys_file_in = get_data_loc(self.data_dir, options.get("SYS_SCALE", "sys_scale.LIST"))
        self.sys_file_out = os.path.join(self.output_dir, "sys_scale.LIST")
        self.chain_dir = os.path.join(self.output_dir, "chains/")
        self.config_dir = os.path.join(self.output_dir, "output")

        self.biascor_dep = self.get_dep(BiasCor, fail=True)
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
                    self.logger.error(f"Done file reported failure. Check output log {self.logfile}")
                    return Task.FINISHED_FAILURE
                else:
                    return Task.FINISHED_SUCCESS
        return 1  # The number of CPUs being utilised

    def calculate_input(self):
        self.logger.debug(f"Calculating input")
        self.set_property("COSMOMC_TEMPLATES", self.template_dir)
        self.set_property("BASEOUTPUT", self.name)
        self.set_property("SYSFILE", self.sys_file_out)
        self.set_property("TOPDIR", self.biascor_dep.output["fit_output_dir"])
        self.set_property("OUTPUTDIR", self.config_dir)
        self.set_property("SUBDIR", self.biascor_dep.output["subdirs"][self.index])
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
            fitopt_scale_overwrites = self.options.get("FITOPT_SCALES", {})
            for label, overwrite in fitopt_scale_overwrites.items():
                for i, line in enumerate(sys_scale):
                    comps = line.split()
                    if label in comps[1]:
                        sys_scale[i] = " ".join(comps[:-1] + [f"{overwrite}"])

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
        }
        final_slurm = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string("\n".join(self.base + sys_scale) + final_slurm)
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
                    a = CreateCov(name, _get_createcov_dir(base_output_dir, stage_number, name), options, [btask], index=i)
                    Task.logger.info(f"Creating createcov task {name} for {btask.name} with {a.num_jobs} jobs")
                    tasks.append(a)

            if len(biascor_tasks) == 0:
                Task.fail_config(f"Create cov task {cname} has no biascor task to run on!")

        return tasks
