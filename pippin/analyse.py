import inspect
import shutil
import subprocess
import os

from pippin.biascor import BiasCor
from pippin.config import mkdirs, get_config, ensure_list
from pippin.cosmomc import CosmoMC
from pippin.snana_fit import SNANALightCurveFit
from pippin.task import Task


class AnalyseChains(Task):  # TODO: Define the location of the output so we can run the lc fitting on it.
    """ Smack the data into something that looks like the simulated data

    CONFIGURATION
    =============

    ANALYSE:
        label:
            MASK_COSMOMC: mask  # partial match
            MASK_BIASCOR: mask # partial match
            HISTOGRAMS: [D_DESSIM, D_DATADES] # Creates histograms based off the input LCFIT_SIMNAME matches. Optional
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
        self.path_to_code_biascor = os.path.dirname(inspect.stack()[0][1]) + "/external/plot_biascor.py"
        self.path_to_code_histogram = os.path.dirname(inspect.stack()[0][1]) + "/external/plot_histogram.py"

        self.covopts = options.get("COVOPTS")
        if isinstance(self.covopts, str):
            self.covopts = [self.covopts]

        self.files = []
        self.names = []
        self.params = []
        self.blind_params = ensure_list(options.get("BLIND", []))

        # Assuming all deps are cosmomc tasks
        self.cosmomc_deps = self.get_deps(CosmoMC)
        self.biascor_deps = self.get_deps(BiasCor)
        self.lcfit_deps = self.get_deps(SNANALightCurveFit)

        self.done_files = []
        if self.cosmomc_deps:
            self.done_files.append(self.done_file)
        if self.biascor_deps:
            self.done_files.append(self.done_file.replace(".txt", "2.txt"))
        if self.lcfit_deps:
            self.done_files.append(self.done_file.replace(".txt", "3.txt"))

        for c in self.cosmomc_deps:
            for covopt in c.output["covopts"]:
                if self.covopts is None or covopt in self.covopts:
                    self.files.append(c.output["base_dict"][covopt])
                    self.names.append(c.output["label"].replace("_", " ") + " " + covopt)
                    for p in c.output["cosmology_params"]:
                        if p not in self.params:
                            self.params.append(p)
        self.logger.debug(f"Analyse task will create plots with {len(self.files)} covopts/plots")

        self.wsummary_files = [b.output["w_summary"] for b in self.biascor_deps]

        self.hubble_plots = list(set([a for c in self.biascor_deps + self.cosmomc_deps for a in c.output.get("hubble_plot")]))
        self.slurm = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=broadwl
#SBATCH --output={log_file}
#SBATCH --account=pi-rkessler
#SBATCH --mem=20GB

cd {output_dir}
python {path_to_code} {files} {output} {blind} {names} {prior} {done_file} {params} {shift}
python {path_to_code_biascor} {wfit_summary} {blind}
python {path_to_histogram} {data_fitres_files} {sim_fitres_files} {types}
"""

    def _check_completion(self, squeue):
        num_success = 0
        for f in self.done_files:
            if os.path.exists(f):
                self.logger.debug(f"Done file found at {f}")
                with open(f) as ff:
                    if "FAILURE" in ff.read():
                        self.logger.error(f"Done file reported failure. Check output log {self.logfile}")
                        return Task.FINISHED_FAILURE
                    else:
                        num_success += 1
        if num_success == len(self.done_files):
            return Task.FINISHED_SUCCESS
        if os.path.exists(self.logfile):
            with open(self.logfile) as f:
                for line in f.read().splitlines():
                    if "kill event" in line:
                        self.logger.error(f"Kill event found in {self.logfile}")
                        self.logger.error(line)
                        return Task.FINISHED_FAILURE
        return 1

    def _run(self, force_refresh):
        data_fitres_files = [os.path.join(l.output["fitres_dirs"][0], l.output["fitopt_map"]["DEFAULT"]) for l in self.lcfit_deps if l.output["is_data"]]
        sim_fitres_files = [os.path.join(l.output["fitres_dirs"][0], l.output["fitopt_map"]["DEFAULT"]) for l in self.lcfit_deps if not l.output["is_data"]]
        types = " ".join([str(a) for l in self.lcfit_deps for a in l.sim_task.output["types_dict"]["IA"]])

        format_dict = {
            "job_name": self.job_name,
            "log_file": self.logfile,
            "done_file": "-d " + os.path.basename(self.done_file),
            "path_to_code": os.path.basename(self.path_to_code),
            "output_dir": self.output_dir,
            "files": " ".join(self.files),
            "output": "-o " + self.name,
            "names": ("-n " + " ".join(['"' + n + '"' for n in self.names])) if self.names else "",
            "blind": ("-b " + " ".join(self.blind_params)) if self.blind_params else "",
            "params": ("-p " + " ".join(self.params)) if self.params else "",
            "shift": "-s" if self.options.get("SHIFT") else "",
            "prior": f"--prior {self.options.get('PRIOR')}" if self.options.get("PRIOR") else "",
            "path_to_code_biascor": os.path.basename(self.path_to_code_biascor),
            "wfit_summary": " ".join(self.wsummary_files),
            "data_fitres_files": "--data" + (" ".join(data_fitres_files)),
            "sim_fitres_files": "--sim" + (" ".join(sim_fitres_files)),
            "types": "--types " + types,
            "path_to_histogram": os.path.basename(self.path_to_code_histogram),
        }
        final_slurm = self.slurm.format(**format_dict)

        new_hash = self.get_hash_from_string(final_slurm)
        old_hash = self.get_old_hash()

        if force_refresh or new_hash != old_hash:
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)
            shutil.copy(self.path_to_code, self.output_dir)
            shutil.copy(self.path_to_code_biascor, self.output_dir)
            shutil.copy(self.path_to_code_histogram, self.output_dir)
            for f in self.hubble_plots:
                self.logger.debug(f"Searching for Hubble plot {f}")
                if f is not None and os.path.exists(f):
                    self.logger.debug(f"Copying Hubble plot {f} to {self.output_dir}")
                    shutil.copy(f, os.path.join(self.output_dir, os.path.basename(f)))
            slurm_output_file = os.path.join(self.output_dir, "slurm.job")
            with open(slurm_output_file, "w") as f:
                f.write(final_slurm)
            self.logger.info(f"Submitting batch job for analyse chains")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")
        return True

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        cosmomc_tasks = Task.get_task_of_type(prior_tasks, CosmoMC)
        biascor_tasks = Task.get_task_of_type(prior_tasks, BiasCor)
        lcfit_tasks = Task.get_task_of_type(prior_tasks, SNANALightCurveFit)

        def _get_analyse_dir(base_output_dir, stage_number, name):
            return f"{base_output_dir}/{stage_number}_ANALYSE/{name}"

        tasks = []
        key = "ANALYSE"
        for cname in c.get(key, []):
            config = c[key].get(cname, {})
            if config is None:
                config = {}
            options = config.get("OPTS", {})

            mask_cosmomc = config.get("MASK_COSMOMC", "")
            mask_biascor = config.get("MASK_BIASCOR", "")
            histograms = config.get("HISTOGRAM", [])

            deps_cosmomc = [c for c in cosmomc_tasks if mask_cosmomc in c.name and mask_biascor in c.output["bcor_name"]]
            deps_biascor = [b for b in biascor_tasks if mask_biascor in b.name]
            deps_hist = [l for l in lcfit_tasks if l.name in histograms]
            if len(histograms) != len(deps_hist):
                Task.fail_config(f"Couldn't match all HISTOGRAM inputs {histograms} with selection: {[l.name for l in lcfit_tasks]}")
            deps = deps_cosmomc + deps_biascor + deps_hist
            if len(deps) == 0:
                Task.fail_config(f"Analyse task {cname} has no dependencies!")

            a = AnalyseChains(cname, _get_analyse_dir(base_output_dir, stage_number, cname), options, deps)
            Task.logger.info(f"Creating Analyse task {cname} for {[c.name for c in deps]} with {a.num_jobs} jobs")
            tasks.append(a)

        return tasks
