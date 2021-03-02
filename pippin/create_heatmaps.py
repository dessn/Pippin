import shutil
import subprocess
import os
import yaml

from pippin.config import mkdirs, get_output_loc, get_config, get_data_loc, read_yaml
from pippin.task import Task
from pippin.snana_sim import SNANASimulation

from pippin.external.SNANA_FITS_to_pd import read_fits

class CreateHeatmaps(Task):
    """ Creates heatmaps from SNANA simulations for SCONE classification

    OUTPUTS:
    ========
    """

    def __init__(self, name, output_dir, config, options, global_config, dependencies=None):
        super().__init__(name, output_dir, config=config, dependencies=dependencies)
        self.options = options
        self.global_config = get_config()

        self.logfile = os.path.join(self.output_dir, "output.log")
        self.conda_env = self.global_config["CreateHeatmaps"]["conda_env"]
        self.output_path = options["output_path"]
        self.path_to_scripts = get_output_loc(self.global_config["CreateHeatmaps"]["location"])
        self.output_dir = output_dir

    def _check_completion(self, squeue):
        #TODO: write a done file
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

    @staticmethod
    def _get_lcdata_paths(sim_dirs):
        sim_paths = [f.path for sim_dir in sim_dirs for f in os.scandir(sim_dirs)]
        lcdata_paths = [path for path in sim_paths if "PHOT" in path]

        return lcdata_paths

    @staticmethod
    def _fitres_to_csv(lcdata_paths, output_dir):
        csv_metadata_paths = []
        csv_lcdata_paths = []

        for path in lcdata_paths:
            csv_metadata_path = os.path.join(output_dir, os.path.basename(path).replace("PHOT.FITS.gz", "HEAD.csv"))
            csv_lcdata_path = os.path.join(output_dir, os.path.basename(path).replace(".FITS.gz", ".csv"))

            metadata, lcdata = read_fits(path)

            metadata.to_csv(csv_metadata_path)
            lcdata.to_csv(csv_lcdata_path)

            csv_metadata_paths.append(csv_metadata_path)
            csv_lcdata_paths.append(csv_lcdata_path)

        return csv_metadata_paths, csv_lcdata_paths

    def _run(self):
        sim_dep = self.get_sim_dependency()
        sim_dirs = sim_dep.output["photometry_dirs"]

        csv_output_dir = os.path.join(self.output_dir, "sim_csvs")
        metadata_paths, lcdata_paths = self._fitres_to_csv(self._get_lcdata_paths(sim_dirs), csv_output_dir)
        script_path = os.path.join(self.path_to_scripts, "create_heatmaps_tfrecord_shellscript.sh")
        config_path = os.path.join(self.output_dir, "create_heatmaps_config.yml")

        self._write_config_file(metadata_paths, lcdata_paths, config_path) # TODO: what if they don't want to train on all sims?

        for i in range(len(metadata_paths)):
            #TODO: determine a good way to estimate time
            cmd = "srun --partition broadwl -N 1 --time 01:00:00 {} {} {} &".format(script_path, config_path, i)
            print(cmd)
            # subprocess.run("module load tensorflow/intel-2.2.0-py37".split(" "))
            subprocess.Popen(cmd.split(" "))

    def get_sim_dependency(self):
        for t in self.dependencies:
            if isinstance(t, SNANASimulation):
                return t.output

        return None

    def _write_config_file(self, metadata_paths, lcdata_paths, config_path):
        config = {}

        config["metadata_paths"] = metadata_paths
        config["lcdata_paths"] = lcdata_paths
        config["output_path"] = self.output_dir
        config["num_wavelength_bins"] = self.config["num_wavelength_bins"] or 32
        config["num_mjd_bins"] = self.config["num_mjd_bins"] or 180
        config["Ia_fraction"] = self.config["Ia_fraction"] or 0.5
        # what to do about sn type ids mapping? keep a default one since read_fits already converts these ids to the ones i'm used to?
        with open(config_path, "w+") as cfgfile:
            cfgfile.write(yaml.dump(config))

    @staticmethod
    def get_tasks(config, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        # TODO: pass in corresponding sim task?
        tasks = []
        for name in config.get("CREATE_HEATMAPS", []): # TODO: is there a way for users to specify global config and not have to write all names?
            output_dir = f"{base_output_dir}/{stage_number}_HEATMAPS/{name}"
            options = config["CREATE_HEATMAPS"][name].get("OPTS")
            if options is None and config["CREATE_HEATMAPS"][name].get("EXTERNAL") is None:
                Task.fail_config(f"CREATE_HEATMAPS task {name} needs to specify OPTS!")
            s = CreateHeatmaps(name, output_dir, config["CREATE_HEATMAPS"][name], options, global_config)
            Task.logger.debug(f"Creating data prep task {name} with {s.num_jobs} jobs, output to {output_dir}")
            tasks.append(s)
        return tasks
