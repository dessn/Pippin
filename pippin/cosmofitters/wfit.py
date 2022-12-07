import inspect
import shutil
import subprocess
import os
from pathlib import Path
import numpy as np

from pippin.config import mkdirs, get_output_loc, get_data_loc, chown_dir, read_yaml
from pippin.create_cov import CreateCov
from pippin.cosmofitters.cosmofit import CosmoFit
from pippin.base import ConfigBasedExecutable
from pippin.task import Task

class WFit(ConfigBasedExecutable, CosmoFit):
    def __init__(self, name, output_dir, create_cov_tasks, config, options, global_config, submit_batch_mode):
        # First check if all required options exist
        # In this case, WFITOPTS must exist with at least 1 entry
        self.submit_batch = submit_batch_mode

        self.wfitopts = options.get("WFITOPTS")
        if self.wfitopts is None:
            Task.fail_config(f"You have not specified any WFITOPTS for task {name}")
        Task.logger.debug(f"WFITOPTS for task {name}: {self.wfitopts}")
        if len(self.wfitopts) == 0:
            Task.fail_config(f"WFITOPTS for task {name} does not have any options!")

        base_file = get_data_loc("wfit/input_file.INPUT")
        super().__init__(name, output_dir, config, base_file, default_assignment=": ", dependencies=create_cov_tasks)
        self.num_jobs = len(self.wfitopts)

        self.create_cov_tasks = create_cov_tasks
        self.logger.debug(f"CreateCov tasks: {self.create_cov_tasks}")
        if self.submit_batch:
            self.create_cov_dirs = []
            for t in self.create_cov_tasks:
                for cov_dir in t.cov_dir:
                    self.create_cov_dirs.append(os.path.join(t.output_dir, "output", cov_dir))
        else:
            self.create_cov_dirs = [os.path.join(t.output_dir, "output") for t in self.create_cov_tasks]
        self.logger.debug(f"CreateCov directories: {self.create_cov_dirs}")
        self.options = options
        self.global_config = global_config
        self.done_file = os.path.join(self.output_dir, "output", "ALL.DONE")
        
        self.job_name = os.path.basename(Path(output_dir).parents[1]) + "_WFIT_" + name
        self.logfile = os.path.join(self.output_dir, "output.log")
        self.input_name = f"{self.job_name}.INPUT"
        self.input_file = os.path.join(self.output_dir, self.input_name)
        
    def _check_completion(self, squeue):
        if os.path.exists(self.done_file):
            self.logger.debug(f"Done file found at {self.done_file}")
            with open(self.done_file) as f:
                if "SUCCESS" in f.read():
                    return Task.FINISHED_SUCCESS
                else:
                    self.logger.error(f"Done file reported failure. Check output log {self.logfile}")
                    self.scan_files_for_error([self.logfile], "ERROR", "EXCEPTION")
                    return Task.FINISHED_FAILURE
        return self.check_for_job(squeue, self.job_name)

    def _run(self):
        self.yaml["CONFIG"]["WFITOPT"] = self.wfitopts
        self.yaml["CONFIG"]["INPDIR"] = self.create_cov_dirs
        self.yaml["CONFIG"]["OUTDIR"] = os.path.join(self.output_dir, "output")
        # Pass all OPTS keys through to the yaml dictionary
        for k, v in self.options.items():
            # Clobber WFITOPTS to WFITOPT
            if k == "WFITOPTS":
                k = "WFITOPT"
            self.yaml["CONFIG"][k] = v
        
        final_output_for_hash = self.get_output_string()

        new_hash = self.get_hash_from_string(final_output_for_hash)

        if self._check_regenerate(new_hash):
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)

            with open(self.input_file, "w") as f:
                f.write(self.get_output_string())

            cmd = ["submit_batch_jobs.sh", os.path.basename(self.input_file)]
            self.logger.debug(f"Submitting wfit job: {' '.join(cmd)} in cwd: {self.output_dir}")
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
        create_cov_tasks = Task.get_task_of_type(prior_tasks, CreateCov)
        submit_batch_mode = False not in [t.submit_batch for t in create_cov_tasks]
        Task.logger.debug(f"wfit submit_batch_mode: {submit_batch_mode}")

        def _get_wfit_dir(base_output_dir, stage_number, name):
            return f"{base_output_dir}/{stage_number}_COSMOFIT/WFIT/{name}"

        tasks = []
        key = "WFIT"
        for name in c.get(key, []):
            config = c[key].get(name, {})
            name = f"WFIT_{name}"
            options = config.get("OPTS", {})

            mask = config.get("MASK", "")

            ctasks = [ctask for ctask in create_cov_tasks if mask in ctask.name]

            t = WFit(name, _get_wfit_dir(base_output_dir, stage_number, name), ctasks, config, options, global_config, submit_batch_mode)
            Task.logger.info(f"Creating WFit task {name} with {t.num_jobs} jobs")
            tasks.append(t)

            if len(create_cov_tasks) == 0:
                Task.fail_config(f"WFit task {name} has no create_cov task to run on!")
        return tasks
