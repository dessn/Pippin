from abc import abstractmethod

from pippin.dataprep import DataPrep
from pippin.task import Task
from pippin.snana_sim import SNANASimulation
from pippin.snana_fit import SNANALightCurveFit


class Classifier(Task):
    """ Classification task

    CONFIGURATION:
    ==============
    CLASSIFICATION:
      label:
        MASK: TEST  # partial match on sim and classifier
        MASK_SIM: TEST  # partial match on sim name
        MASK_FIT: TEST  # partial match on lcfit name
        MODE: train/predict # Some classifiers dont need training and so you can set to predict straight away
        OPTS:
          CHANGES_FOR_INDIVIDUAL_CLASSIFIERS

    OUTPUTS:
    ========
        name : name given in the yml
        output_dir: top level output directory
        prob_column_name: name of the column to get probabilities out of
        predictions_filename: location of csv filename with id/probs

    """

    TRAIN = 0
    PREDICT = 1

    def __init__(self, name, output_dir, dependencies, mode, options):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.options = options
        self.mode = mode
        self.output["prob_column_name"] = self.get_prob_column_name()

    @abstractmethod
    def predict(self, force_refresh):
        """ Predict probabilities for given dependencies

        :param force_refresh: to force refresh and rerun - do not pass hash checks
        :return: true or false for success in launching the job
        """
        pass

    @abstractmethod
    def train(self, force_refresh):
        """ Train a model to file for given dependencies

        :param force_refresh: to force refresh and rerun - do not pass hash checks
        :return: true or false for success in launching the job
        """
        pass

    @staticmethod
    def get_requirements(config):
        """ Return what data is actively used by the classifier

        :param config: the input dictionary `OPTS` from the config file
        :return: a two tuple - (needs simulation photometry, needs a fitres file)
        """
        return True, True

    def get_fit_dependency(self, output=True):
        for t in self.dependencies:
            if isinstance(t, SNANALightCurveFit):
                return t.output if output else t
        return None

    def get_simulation_dependency(self):
        for t in self.dependencies:
            if isinstance(t, SNANASimulation) or isinstance(t, DataPrep):
                return t
        for t in self.get_fit_dependency(output=False).dependencies:
            if isinstance(t, SNANASimulation) or isinstance(t, DataPrep):
                return t
        return None

    def get_model_classifier(self):
        for t in self.dependencies:
            if isinstance(t, Classifier):
                return t
        return None

    def _run(self, force_refresh):
        if self.mode == Classifier.TRAIN:
            return self.train(force_refresh)
        elif self.mode == Classifier.PREDICT:
            return self.predict(force_refresh)

    def get_unique_name(self):
        name = self.name
        use_sim, use_fit = self.get_requirements(self.options)
        if use_fit:
            name += "_" + self.get_fit_dependency()["name"]
        else:
            name += "_" + self.get_simulation_dependency().output["name"]
        return name

    def get_prob_column_name(self):
        m = self.get_model_classifier()
        if m is None:
            return f"PROB_{self.get_unique_name()}"
        else:
            return m.output["prob_column_name"]

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        from pippin.classifiers.factory import ClassifierFactory

        def _get_clas_output_dir(base_output_dir, stage_number, sim_name, fit_name, clas_name, extra=None):
            fit_name = "" if fit_name is None else "_" + fit_name
            sim_name = "" if sim_name is None else "_" + sim_name
            extra_name = "" if extra is None else "_" + extra
            return f"{base_output_dir}/{stage_number}_CLAS/{clas_name}{sim_name}{fit_name}{extra_name}"

        tasks = []
        lcfit_tasks = Task.get_task_of_type(prior_tasks, SNANALightCurveFit)
        sim_tasks = Task.get_task_of_type(prior_tasks, DataPrep, SNANASimulation)
        for clas_name in c.get("CLASSIFICATION", []):
            config = c["CLASSIFICATION"][clas_name]
            name = config["CLASSIFIER"]
            cls = ClassifierFactory.get(name)
            options = config.get("OPTS", {})
            mode = config["MODE"].lower()
            assert mode in ["train", "predict"], "MODE should be either train or predict"
            if mode == "train":
                mode = Classifier.TRAIN
            else:
                mode = Classifier.PREDICT

            needs_sim, needs_lc = cls.get_requirements(options)

            runs = []
            if needs_sim and needs_lc:
                runs = [(l.dependencies[0], l) for l in lcfit_tasks]
            elif needs_sim:
                runs = [(s, None) for s in sim_tasks]
            elif needs_lc:
                runs = [(l.dependencies[0], l) for l in lcfit_tasks]
            else:
                Task.logger.warn(f"Classifier {name} does not need sims or fits. Wat.")

            num_gen = 0
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_fit = config.get("MASK_FIT", "")
            for s, l in runs:
                sim_name = s.name if s is not None else None
                fit_name = l.name if l is not None else None
                matched_sim = sim_name is None
                matched_fit = fit_name is None
                if mask:
                    matched_sim = matched_sim or mask in sim_name
                if mask_sim:
                    matched_sim = matched_sim or mask_sim in sim_name
                if mask:
                    matched_fit = matched_fit or mask in sim_name
                if mask_fit:
                    matched_fit = matched_fit or mask_sim in sim_name
                if not matched_fit or not matched_sim:
                    continue
                deps = []
                if s is not None:
                    deps.append(s)
                if l is not None:
                    deps.append(l)

                model = options.get("MODEL")
                if model is not None:
                    for t in tasks:
                        if model == t.name:
                            # deps.append(t)
                            extra = t.get_unique_name()
                            clas_output_dir = _get_clas_output_dir(base_output_dir, stage_number, sim_name, fit_name, clas_name, extra=extra)
                            cc = cls(clas_name, clas_output_dir, deps + [t], mode, options)
                            Task.logger.info(f"Creating classification task {name} with {cc.num_jobs} jobs, for LC fit {fit_name} on simulation {sim_name}")
                            num_gen += 1
                            tasks.append(cc)
                else:
                    clas_output_dir = _get_clas_output_dir(base_output_dir, stage_number, sim_name, fit_name, clas_name)
                    cc = cls(clas_name, clas_output_dir, deps, mode, options)
                    Task.logger.info(f"Creating classification task {name} with {cc.num_jobs} jobs, for LC fit {fit_name} on simulation {sim_name}")
                    num_gen += 1
                    tasks.append(cc)
            if num_gen == 0:
                Task.logger.error(f"Classifier {name} with masks |{mask}|{mask_sim}|{mask_fit}| matched no combination of sims and fits")
                return None  # This should cause pippin to crash, which is probably what we want
        return tasks
