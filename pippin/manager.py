import os
import inspect

from pippin.classifiers.factory import ClassifierFactory
from pippin.config import get_logger, get_config
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_simulation import SNANASimulation


class Manager:
    def __init__(self, filename, config):
        self.logger = get_logger()
        self.filename = filename
        self.run_config = config
        self.global_config = get_config()

        self.prefix = "PIPPIN_" + filename
        self.output_dir = None

    def _get_sim_output_dir(self, sim_name):
        return f"{self.output_dir}/0_SIM/{self.prefix}_{sim_name}"

    def _get_lc_output_dir(self, sim_name, fit_name):
        return f"{self.output_dir}/1_LCFIT/{self.prefix}_{sim_name}_{fit_name}"

    def _get_clas_output_dir(self, sim_name, fit_name, clas_name):
        return f"{self.output_dir}/2_CLAS/{self.prefix}_{sim_name}_{fit_name}_{clas_name}"

    def execute(self):
        self.logger.info(f"Executing pipeline for prefix {self.prefix}")

        self.output_dir = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + "/../" + self.global_config['OUTPUT']['output_dir'] + "/" + self.filename)
        self.logger.info(f"Output will be located in {self.output_dir}")
        c = self.run_config

        num_sims = len(c["SIM"].keys())
        num_fits = len(c["LCFIT"].keys())
        num_clas = len(c["CLASSIFICATION"].keys())
        self.logger.info(f"Found {num_sims} simulation(s), {num_fits} LC fit(s), {num_clas} classifiers")
        self.logger.info("")
        self.logger.info("")

        sim_hashes = {}
        for sim_name in c["SIM"]:
            sim_output_dir = self._get_sim_output_dir(sim_name)
            self.logger.debug(f"Running simulation {sim_name}, output to {sim_output_dir}")
            s = SNANASimulation(sim_output_dir, f"{self.prefix}_{sim_name}", c["SIM"][sim_name], self.global_config)
            sim_hash = s.run()
            if not sim_hash:
                exit(1)
            sim_hashes[sim_name] = sim_hash
            self.logger.info("")

        self.logger.info("Completed all simulations")
        self.logger.info("")
        self.logger.info("")

        lc_hashes = {}
        for sim_name in c["SIM"]:
            for fit_name in c["LCFIT"]:
                fit_output_dir = self._get_lc_output_dir(sim_name, fit_name)
                self.logger.info(f"Fitting {fit_name} for simulation {sim_name}")
                f = SNANALightCurveFit(fit_output_dir, f"{self.prefix}_{sim_name}", c["LCFIT"][fit_name], self.global_config, sim_hashes[sim_name])
                lc_hash = f.run()
                if not lc_hash:
                    exit(1)
                lc_hashes[f"{sim_name}_{fit_name}"] = lc_hash
                self.logger.info("")

        self.logger.info("Completed all light curve fitting")
        self.logger.info("")
        self.logger.info("")

        classifier_hashes = {}
        for sim_name in c["SIM"]:
            for fit_name in c["LCFIT"]:
                for clas_name in c["CLASSIFICATION"]:
                    clas_output_dir = self._get_clas_output_dir(sim_name, fit_name, clas_name)
                    self.logger.info(f"Classifying {clas_name} for LC fit {fit_name} on simulation {sim_name}")

                    config = c["CLASSIFICATION"][clas_name]
                    options = config.get("OPTS", {})
                    name = config["CLASSIFIER"]
                    self.logger.debug(f"Attempting to initialise class {name}")
                    cls = ClassifierFactory.get(name)
                    cc = cls(self._get_sim_output_dir(sim_name), self._get_lc_output_dir(sim_name, fit_name) + f"/output/{self.prefix}_{sim_name}", clas_output_dir, options)
                    clas_hash = cc.classify()
                    if not clas_hash:
                        exit(1)
                    classifier_hashes[f"{sim_name}_{fit_name}_{clas_name}"] = clas_hash
                    self.logger.info("")