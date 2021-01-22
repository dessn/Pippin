import inspect
import json
import shutil
import subprocess
import os
from pathlib import Path
import numpy as np

from pippin.biascor import BiasCor
from pippin.config import mkdirs, get_config, ensure_list, get_data_loc
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
            MASK_LCFIT: [D_DESSIM, D_DATADES] # Creates histograms based off the input LCFIT_SIMNAME matches. Optional
            EFFICIENCY: [D_DESSIM, D_DATADES] # Attempts to make histogram efficiency
            OPTS:
                BLIND: [omegam, w]  # Optional. Parameters to blind. Single or list. omegam, omegal, w, wa, nu, etc
                COVOPTS: [ALL, NOSYS] # Optional. Covopts to match. Single or list. Exact match.
                ADDITIONAL_SCRIPTS:
                  - path/to/your/script.py
                  - anotherscript.py

    OUTPUTS
    =======



    """

    def __init__(self, name, output_dir, config, options, dependencies=None):
        super().__init__(name, output_dir, config=config, dependencies=dependencies)
        self.options = options
        self.global_config = get_config()

        self.logfile = os.path.join(self.output_dir, "output.log")

        self.job_name = os.path.basename(Path(output_dir).parents[1]) + "_ANALYSE_" + os.path.basename(output_dir)

        self.path_to_codes = []
        self.done_files = []

        self.plot_code_dir = os.path.join(os.path.dirname(inspect.stack()[0][1]), "external")

        self.covopts = options.get("COVOPTS")
        self.singular_blind = options.get("SINGULAR_BLIND", False)
        if isinstance(self.covopts, str):
            self.covopts = [self.covopts]

        self.cosmomc_input_files = []
        self.cosmomc_output_files = []
        self.cosmomc_covopts = []
        self.names = []
        self.params = []

        # Assuming all deps are cosmomc tasks
        self.cosmomc_deps = self.get_deps(CosmoMC)
        self.blind = np.any([c.output["blind"] for c in self.cosmomc_deps])
        if self.blind:
            self.blind_params = ["w", "om", "ol", "omegam", "omegal"]
        else:
            self.blind_params = []
        self.biascor_deps = self.get_deps(BiasCor)
        self.lcfit_deps = self.get_deps(SNANALightCurveFit)

        if self.cosmomc_deps:
            self.add_plot_script_to_run("parse_cosmomc.py")
            self.add_plot_script_to_run("plot_cosmomc.py")
            self.add_plot_script_to_run("plot_errbudget.py")
        if self.biascor_deps:
            self.add_plot_script_to_run("parse_biascor.py")
            self.add_plot_script_to_run("plot_biascor.py")
        if self.lcfit_deps:
            self.add_plot_script_to_run("parse_lcfit.py")
            self.add_plot_script_to_run("plot_histogram.py")
            self.add_plot_script_to_run("plot_efficiency.py")

        if self.options.get("ADDITIONAL_SCRIPTS") is not None:
            vals = ensure_list(self.options.get("ADDITIONAL_SCRIPTS"))
            for v in vals:
                self.add_plot_script_to_run(v)

        self.done_file = self.done_files[-1]

        for c in self.cosmomc_deps:
            for covopt in c.output["covopts"]:
                self.cosmomc_input_files.append(c.output["base_dict"][covopt])
                self.cosmomc_output_files.append(c.output["label"] + "_" + covopt + ".csv.gz")
                self.cosmomc_covopts.append(covopt)
                self.names.append(c.output["label"].replace("_", " ") + " " + covopt)
                for p in c.output["cosmology_params"]:
                    if p not in self.params:
                        self.params.append(p)
            self.logger.debug(f"Analyse task will create CosmoMC plots with {len(self.cosmomc_input_files)} covopts/plots")

        self.wsummary_files = [b.output["w_summary"] for b in self.biascor_deps]

        # Get the fitres and m0diff files we'd want to parse for Hubble diagram plotting
        self.biascor_fitres_input_files = [os.path.join(m, "FITOPT000_MUOPT000.FITRES.gz") for b in self.biascor_deps for m in b.output["m0dif_dirs"]]
        self.biascor_prob_col_names = [b.output["prob_column_name"] for b in self.biascor_deps for m in b.output["m0dif_dirs"]]
        self.biascor_fitres_output_files = [
            b.name + "__" + os.path.basename(m).replace("OUTPUT_BBCFIT", "1") + "__FITOPT0_MUOPT0.fitres.gz"
            for b in self.biascor_deps
            for m in b.output["m0dif_dirs"]
        ]

        self.biascor_m0diffs = []
        self.biascor_m0diff_output = "all_biascor_m0diffs.csv"
        self.biascor_fitres_combined = "all_biascor_fitres.csv.gz"

        self.slurm = """{sbatch_header}
cd {output_dir}

"""

    def get_slurm_raw(self):
        base = self.slurm
        template = """
echo "\nExecuting {path}"
python {path} {{input_yml}}
if [ $? -ne 0 ]; then
    echo FAILURE > {donefile}
else
    echo SUCCESS > {donefile}
fi
"""
        for path, donefile in zip(self.path_to_codes, self.done_files):
            base += template.format(path=os.path.basename(path), donefile=donefile)
        return base

    def add_plot_script_to_run(self, script_name):
        script_path = get_data_loc(script_name, extra=self.plot_code_dir)
        if script_path is None:
            self.fail_config(f"Cannot resolve script {script_name} relative to {self.plot_code_dir}. Please use a variable or abs path.")
        else:
            self.logger.debug(f"Adding script path {script_path} to plotting code.")
        self.path_to_codes.append(script_path)
        self.done_files.append(os.path.join(self.output_dir, os.path.basename(script_name).split(".")[0] + ".done"))

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
        return self.check_for_job(squeue, self.job_name)

    def _run(self):

        # Get the m0diff files for everything
        for b in self.biascor_deps:
            for m in b.output["m0dif_dirs"]:
                self.logger.info(f"Looking at M0diff dir {m}")
                sim_number = 1
                if os.path.basename(m).isdigit():
                    sim_number = int(os.path.basename(m))
                files = [f for f in sorted(os.listdir(m)) if f.endswith(".M0DIF") or f.endswith(".M0DIF.gz")]
                for f in files:
                    muopt_num = int(f.split("MUOPT")[-1].split(".")[0])
                    fitopt_num = int(f.split("FITOPT")[-1].split("_")[0])
                    if muopt_num == 0:
                        muopt = "DEFAULT"
                    else:
                        muopt = b.output["muopts"][muopt_num - 1]  # Because 0 is default

                    if fitopt_num == 0:
                        fitopt = "DEFAULT"
                    else:
                        fitopt = b.output["fitopt_index"][fitopt_num]

                    self.biascor_m0diffs.append((b.name, sim_number, muopt, muopt_num, fitopt, fitopt_num, os.path.join(m, f)))

        data_fitres_files = [os.path.join(l.output["fitres_dirs"][0], l.output["fitopt_map"]["DEFAULT"]) for l in self.lcfit_deps if l.output["is_data"]]
        data_fitres_output = [d.split("/")[-4] + ".csv.gz" for d in data_fitres_files]
        sim_fitres_files = [os.path.join(l.output["fitres_dirs"][0], l.output["fitopt_map"]["DEFAULT"]) for l in self.lcfit_deps if not l.output["is_data"]]
        sim_fitres_output = [d.split("/")[-4] + ".csv.gz" for d in sim_fitres_files]
        types = list(set([a for l in self.lcfit_deps for a in l.sim_task.output["types_dict"]["IA"]]))
        input_yml_file = "input.yml"
        output_dict = {
            "COSMOMC": {
                "INPUT_FILES": self.cosmomc_input_files,
                "PARSED_FILES": self.cosmomc_output_files,
                "PARSED_COVOPTS": self.cosmomc_covopts,
                "PARAMS": self.params,
                "SHIFT": self.options.get("SHIFT", False),
                "PRIOR": self.options.get("PRIOR"),
                "NAMES": self.names,
                "CONTOUR_COVOPTS": self.covopts,
                "SINGULAR_BLIND": self.singular_blind,
            },
            "BIASCOR": {
                "WFIT_SUMMARY_INPUT": self.wsummary_files,
                "WFIT_SUMMARY_OUTPUT": "all_biascor.csv",
                "FITRES_INPUT": self.biascor_fitres_input_files,
                "FITRES_PROB_COLS": self.biascor_prob_col_names,
                "FITRES_PARSED": self.biascor_fitres_output_files,
                "M0DIFF_INPUTS": self.biascor_m0diffs,
                "M0DIFF_PARSED": self.biascor_m0diff_output,
                "FITRES_COMBINED": self.biascor_fitres_combined,
            },
            "OUTPUT_NAME": self.name,
            "BLIND": self.blind_params,
            "LCFIT": {
                "DATA_FITRES_INPUT": data_fitres_files,
                "SIM_FITRES_INPUT": sim_fitres_files,
                "DATA_FITRES_PARSED": data_fitres_output,
                "SIM_FITRES_PARSED": sim_fitres_output,
                "IA_TYPES": types,
            },
        }
        header_dict = {
                    "job-name": self.job_name,
                    "time": "1:00:00",
                    "ntasks": "1",
                    "cpus-per-task": "1",
                    "output": self.logfile,
                    "mem-per-cpu": "20"
                } 
        if self.gpu:
            self.sbatch_header = self.sbatch_gpu_header
        else:
            self.sbatch_header = self.sbatch_cpu_header
        self.update_header(header_dict)
        format_dict = {"sbatch_header": self.sbatch_header, "output_dir": self.output_dir, "input_yml": input_yml_file}
        final_slurm = self.get_slurm_raw().format(**format_dict)

        new_hash = self.get_hash_from_string(final_slurm + json.dumps(output_dict))

        if self._check_regenerate(new_hash):
            self.logger.debug("Regenerating and launching task")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            mkdirs(self.output_dir)
            self.save_new_hash(new_hash)
            for c in self.path_to_codes:
                shutil.copy(c, self.output_dir)
            input_yml_path = os.path.join(self.output_dir, input_yml_file)
            with open(input_yml_path, "w") as f:
                json.dump(output_dict, f, indent=2)
                self.logger.debug(f"Input yml file written out to {input_yml_path}")

            slurm_output_file = os.path.join(self.output_dir, "slurm.job")
            with open(slurm_output_file, "w") as f:
                f.write(final_slurm)
            self.logger.info(f"Submitting batch job for analyse chains")
            subprocess.run(["sbatch", slurm_output_file], cwd=self.output_dir)
        else:
            self.logger.info("Hash check passed, not rerunning")
        return True

    @staticmethod
    def get_tasks(configs, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        def _get_analyse_dir(base_output_dir, stage_number, name):
            return f"{base_output_dir}/{stage_number}_ANALYSE/{name}"

        tasks = []
        key = "ANALYSE"
        for cname in configs.get(key, []):
            config = configs[key].get(cname, {})
            if config is None:
                config = {}
            options = config.get("OPTS", {})

            mask_cosmomc = config.get("MASK_COSMOMC")
            mask_biascor = config.get("MASK_BIASCOR")
            if config.get("HISTOGRAM") is not None:
                Task.fail_config("Sorry to do this, but please change HISTOGRAM into MASK_LCFIT to bring it into line with others.")
            mask_lcfit = config.get("MASK_LCFIT")
            # TODO: Add aggregation to compile all the plots here

            deps_cosmomc = Task.match_tasks_of_type(mask_cosmomc, prior_tasks, CosmoMC, match_none=False)
            deps_biascor = Task.match_tasks_of_type(mask_biascor, prior_tasks, BiasCor, match_none=False)
            deps_lcfit = Task.match_tasks_of_type(mask_lcfit, prior_tasks, SNANALightCurveFit, match_none=False)

            deps = deps_cosmomc + deps_biascor + deps_lcfit
            if len(deps) == 0:
                Task.fail_config(f"Analyse task {cname} has no dependencies!")

            a = AnalyseChains(cname, _get_analyse_dir(base_output_dir, stage_number, cname), config, options, deps)
            Task.logger.info(f"Creating Analyse task {cname} for {[c.name for c in deps]} with {a.num_jobs} jobs")
            tasks.append(a)

        return tasks
