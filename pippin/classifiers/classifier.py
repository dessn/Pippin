import os
from abc import abstractmethod

from pippin.config import get_output_loc, get_data_loc
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
        MASK: TEST  # partial match on sim name and lcfit name
        MASK_SIM: TEST  # partial match on sim name
        MASK_FIT: TEST  # partial match on lcfit name
        COMBINE_MASK: TEST1,TEST2 # combining multiple masks (e.g. SIM_Ia,SIM_CC) - *exact* match on sim name and lcfit name

        MODE: train/predict # Some classifiers dont need training and so you can set to predict straight away
        OPTS:
          # Masks for optional dependencies. Since whether an optional dependency is allowed is classifier specific, this is a classifier opt
          # If any are defined, they take precedence, otherwise use above masks for optional dependencies too
          OPTIONAL_MASK: TEST
          OPTIONAL_MASK_SIM: TEST
          OPTIONAL_MASK_FIT: TEST

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

    def __init__(self, name, output_dir, config, dependencies, mode, options, index=0, model_name=None):
        super().__init__(name, output_dir, config=config, dependencies=dependencies)
        self.options = options
        self.index = index
        self.mode = mode
        self.model_name = model_name
        self.output["prob_column_name"] = self.get_prob_column_name()
        self.output["index"] = index

    @abstractmethod
    def predict(self):
        """ Predict probabilities for given dependencies

        :return: true or false for success in launching the job
        """
        pass

    @abstractmethod
    def train(self):
        """ Train a model to file for given dependencies

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


    @staticmethod
    def get_optional_requirements(config):
        """ Return what data *may* be used by the classier.
        Default behaviour:
            if OPTIONAL_MASK != "":
                True, True
            if OPTIONAL_MASK_SIM != "":
                True, False
            if OPTIONAL_MASK_FIT != "":
                False, True


        :param config: the input dictionary `OPTS` from the config file
        :return: a two tuple - (needs simulation photomerty, needs a fitres file)
        """

        opt_sim = ("OPTIONAL_MASK" in config) or ("OPTIONAL_MASK_SIM" in config)
        opt_fit = ("OPTIONAL_MASK" in config) or ("OPTIONAL_MASK_FIT" in config)
        return opt_sim, opt_fit 

    def get_fit_dependency(self, output=True):
        fit_deps = []
        for t in self.dependencies:
            if isinstance(t, SNANALightCurveFit):
                fit_deps.append(t.output) if output else fit_deps.append(t)
        return fit_deps

    def get_simulation_dependency(self):
        sim_deps = []
        for t in self.dependencies:
            if isinstance(t, SNANASimulation) or isinstance(t, DataPrep):
                sim_deps.append(t)
        for t in self.get_fit_dependency(output=False):
            for dep in t.dependencies:
                if isinstance(t, SNANASimulation) or isinstance(t, DataPrep):
                    sim_deps.append(t)
        return sim_deps

    def validate_model(self):
        if self.mode == Classifier.PREDICT:
            model = self.options.get("MODEL")
            if model is None:
                Task.fail_config(f"Classifier {self.name} is in predict mode but does not have a model specified")
            model_classifier = self.get_model_classifier()
            if model_classifier is not None and model_classifier.name == model:
                return True
            path = get_data_loc(model)
            if not os.path.exists(path):
                Task.fail_config(f"Classifier {self.name} does not have a classifier dependency and model is not a serialised file path")
        return True

    def get_model_classifier(self):
        for t in self.dependencies:
            if isinstance(t, Classifier):
                return t
        return None

    def _run(self):
        if self.mode == Classifier.TRAIN:
            return self.train()
        elif self.mode == Classifier.PREDICT:
            return self.predict()

    def get_unique_name(self):
        name = self.name
        use_sim, use_fit = self.get_requirements(self.options)
        if use_fit:
            for t in self.get_fit_dependency():
                name += f"_{t['name']}"
        else:
            for t in self.get_simulation_dependency():
                name += f"_{t.output['name']}"
        return name

    def get_prob_column_name(self):
        m = self.get_model_classifier()
        if m is None:
            if self.model_name is not None:
                return f"PROB_{self.model_name}"
            else:
                return f"PROB_{self.get_unique_name()}"
        else:
            return m.output["prob_column_name"]

    @staticmethod
    def get_tasks(c, prior_tasks, base_output_dir, stage_number, prefix, global_config):
        from pippin.classifiers.factory import ClassifierFactory

        def _get_clas_output_dir(base_output_dir, stage_number, sim_name, fit_name, clas_name, index=None, extra=None):
            sim_name = "" if sim_name is None or fit_name is not None else "_" + sim_name
            fit_name = "" if fit_name is None else "_" + fit_name
            extra_name = "" if extra is None else "_" + extra
            index = "" if index is None else f"_{index}"
            return f"{base_output_dir}/{stage_number}_CLAS/{clas_name}{index}{sim_name}{fit_name}{extra_name}"

        def get_num_ranseed(sim_tasks, lcfit_tasks):
            num = 0
            if len(sim_tasks) > 0:
                return min([len(sim_task.output["sim_folders"]) for sim_task in sim_tasks])
            if len(lcfit_tasks) > 0:
                return min([len(lcfit_task.output["fitres_dirs"]) for lcfit_task in lcfit_tasks])
            raise ValueError("Classifier dependency has no sim_task or lcfit_task?")

        tasks = []
        lcfit_tasks = Task.get_task_of_type(prior_tasks, SNANALightCurveFit)
        sim_tasks = Task.get_task_of_type(prior_tasks, DataPrep, SNANASimulation)
        for clas_name in c.get("CLASSIFICATION", []):
            config = c["CLASSIFICATION"][clas_name]
            name = config["CLASSIFIER"]
            cls = ClassifierFactory.get(name)
            options = config.get("OPTS", {})
            if options == None:
                Task.fail_config(f"Classifier {clas_name} has no OPTS specified -- either remove the OPTS keyword or specify some options under it")
            if "MODE" not in config:
                Task.fail_config(f"Classifier task {clas_name} needs to specify MODE as train or predict")
            mode = config["MODE"].lower()
            assert mode in ["train", "predict"], "MODE should be either train or predict"
            if mode == "train":
                mode = Classifier.TRAIN
            else:
                mode = Classifier.PREDICT

            # Prevent mode = predict and SIM_FRACTION < 1
            if mode == Classifier.PREDICT and options.get("SIM_FRACTION", 1) > 1:
                Task.fail_config("SIM_FRACTION must be 1 (all sims included) for predict mode")

            # Validate that train is not used on certain classifiers
            if mode == Classifier.TRAIN:
                assert name not in ["PerfectClassifier", "UnityClassifier", "FitProbClassifier"], f"Can not use train mode with {name}"

            needs_sim, needs_lc = cls.get_requirements(options)

            # Load in all optional tasks
            opt_sim, opt_lc = cls.get_optional_requirements(options)
            opt_deps = []
            if opt_sim or opt_lc:
                # Get all optional masks
                mask = options.get("OPTIONAL_MASK", "") 
                mask_sim = options.get("OPTIONAL_MASK_SIM", "")
                mask_fit = options.get("OPTIONAL_MASK_FIT", "")
                
                # If no optional masks are set, use base masks
                if not any([mask, mask_sim, mask_fit]):
                    mask = config.get("MASK", "")
                    mask_sim = config.get("MASK_SIM", "")
                    mask_fit = config.get("MASK_FIT", "")

                # Get optional sim tasks
                optional_sim_tasks = []
                if opt_sim:
                    if not any([mask, mask_sim]):
                        Task.logger.debug(f"No optional sim masks set, all sim tasks included as dependendencies")
                        optional_sim_tasks = sim_tasks
                    else:
                        for s in sim_tasks:
                            if mask_sim and mask_sim in s.name:
                                optional_sim_tasks.append(s)
                            elif mask and mask in s.name:
                                optional_sim_tasks.append(s)
                        if len(optional_sim_tasks) == 0:
                            Task.logger.warn(f"Optional SIM dependency but no matching sim tasks for MASK: {mask} or MASK_SIM: {mask_sim}")
                        else:
                            Task.logger.debug(f"Found {len(optional_sim_tasks)} optional SIM dependencies")
                # Get optional lcfit tasks
                optional_lcfit_tasks = []
                if opt_lc:
                    if not any([mask, mask_fit]):
                        Task.logger.debug(f"No optional lcfit masks set, all lcfit tasks included as dependendencies")
                        optional_lcfit_tasks = lcfit_tasks
                    else:
                        for l in lcfit_tasks:
                            if mask_fit and mask_fit in l.name:
                                optional_lcfit_tasks.append(l)
                            elif mask and mask in l.name:
                                optional_lcfit_tasks.append(l)
                            if len(optional_lcfit_tasks) == 0:
                                Task.logger.warn(f"Optional LCFIT dependency but no matching lcfit tasks for MASK: {mask} or MASK_FIT: {mask_fit}")
                            else:
                                Task.logger.debug(f"Found {len(optional_lcfit_tasks)} optional LCFIT dependencies")
                opt_deps = optional_sim_tasks + optional_lcfit_tasks

            runs = []
            if "COMBINE_MASK" in config:
                combined_tasks = []
                regular_tasks = []
                if needs_lc:
                    for l in lcfit_tasks:
                        if l is not None and l.name in config["COMBINE_MASK"]:
                            combined_tasks.append((l.dependencies[0], l))
                        if l is not None and l.name not in config["COMBINE_MASK"]:
                            regular_tasks.append([(l.dependencies[0], l)])
                else:
                    for s in sim_tasks:
                        if s is not None and s.name in config["COMBINE_MASK"]:
                            combined_tasks.append((s, None))
                        if s is not None and s.name not in config["COMBINE_MASK"]:
                            regular_tasks.append([(s, None)])
                runs = [combined_tasks] + regular_tasks
            else:
                if needs_lc:
                    runs = [[(l.dependencies[0], l)] for l in lcfit_tasks]
                elif needs_sim:
                    runs = [[(s, None)] for s in sim_tasks]
                else:
                    Task.logger.warn(f"Classifier {name} does not need sims or fits. Wat.")

            num_gen = 0
            mask = config.get("MASK", "")
            mask_sim = config.get("MASK_SIM", "")
            mask_fit = config.get("MASK_FIT", "")
            mask_combined = config.get("COMBINE_MASK", "")
            for run in runs:
                matched_sim = True
                matched_fit = True
                matched_combined = True
                if mask_combined:
                    matched_combined = len(run) > 1
                else:
                    if len(run) > 1:
                        Task.logger.warn(f"Classifier {name} has multiple tasks -- this should only occur when COMBINE_MASK is specified. Using first task.")

                    s, l = run[0]
                    sim_name = s.name if s is not None else None
                    fit_name = l.name if l is not None else None
                    if mask:
                        matched_sim = matched_sim and mask in sim_name
                    if mask_sim:
                        matched_sim = matched_sim and mask_sim in sim_name
                    if mask:
                        matched_fit = matched_fit and mask in sim_name
                    if mask_fit:
                        matched_fit = matched_fit and mask_sim in sim_name
                if not matched_fit or not matched_sim or not matched_combined:
                    continue
                sim_deps = [sim_fit_tuple[0] for sim_fit_tuple in run if sim_fit_tuple[0] is not None]
                fit_deps = [sim_fit_tuple[1] for sim_fit_tuple in run if sim_fit_tuple[1] is not None]

                model = options.get("MODEL")

                # Validate to make sure training samples only have one sim.
                if mode == Classifier.TRAIN:
                    for s in sim_deps:
                        if s is not None:
                            folders = s.output["sim_folders"]
                            assert (
                                len(folders) == 1
                            ), f"Training requires one version of the sim, you have {len(folders)} for sim task {s}. Make sure your training sim doesn't set RANSEED_CHANGE"
                    for l in fit_deps:
                        if l is not None:
                            folders = l.output["fitres_dirs"]
                            assert (
                                len(folders) == 1
                            ), f"Training requires one version of the lcfits, you have {len(folders)} for lcfit task {l}. Make sure your training sim doesn't set RANSEED_CHANGE"

                deps = sim_deps + fit_deps + opt_deps

                sim_name = "_".join([s.name for s in sim_deps if s is not None]) if len(sim_deps) > 0 else None
                fit_name = "_".join([l.name for l in fit_deps if l is not None]) if len(fit_deps) > 0 else None

                if model is not None:
                    if "/" in model or "." in model:
                        potential_path = get_output_loc(model)
                        if os.path.exists(potential_path):
                            extra = os.path.basename(os.path.dirname(potential_path))

                            # Nasty duplicate code, TODO fix this
                            indexes = get_num_ranseed(sim_deps, fit_deps)
                            for i in range(indexes):
                                num = i + 1 if indexes > 1 else None
                                clas_output_dir = _get_clas_output_dir(base_output_dir, stage_number, sim_name, fit_name, clas_name, index=num, extra=extra)
                                cc = cls(clas_name, clas_output_dir, config, deps, mode, options, index=i, model_name=extra)
                                Task.logger.info(
                                    f"Creating classification task {name} with {cc.num_jobs} jobs, for LC fit {fit_name} on simulation {sim_name} and index {i}"
                                )
                                num_gen += 1
                                tasks.append(cc)

                        else:
                            Task.fail_config(f"Your model {model} looks like a path, but I couldn't find a model at {potential_path}")
                    else:
                        if len(tasks) == 0:
                            Task.fail_config(f"Your model {model} has not yet been defined.")
                        for t in tasks:
                            if model == t.name:
                                # deps.append(t)
                                extra = t.get_unique_name()

                                assert t.__class__ == cls, f"Model {clas_name} with class {cls} has model {model} with class {t.__class__}, they should match!"

                                indexes = get_num_ranseed(sim_deps, fit_deps)
                                for i in range(indexes):
                                    num = i + 1 if indexes > 1 else None
                                    clas_output_dir = _get_clas_output_dir(base_output_dir, stage_number, sim_name, fit_name, clas_name, index=num, extra=extra)
                                    cc = cls(clas_name, clas_output_dir, config, deps + [t], mode, options, index=i)
                                    Task.logger.info(
                                        f"Creating classification task {name} with {cc.num_jobs} jobs, for LC fit {fit_name} on simulation {sim_name} and index {i}"
                                    )
                                    num_gen += 1
                                    tasks.append(cc)
                else:

                    indexes = get_num_ranseed(sim_deps, fit_deps)
                    for i in range(indexes):
                        num = i + 1 if indexes > 1 else None
                        clas_output_dir = _get_clas_output_dir(base_output_dir, stage_number, sim_name, fit_name, clas_name, index=num)
                        print(clas_output_dir)
                        print(deps)
                        cc = cls(clas_name, clas_output_dir, config, deps, mode, options, index=i)
                        Task.logger.info(
                            f"Creating classification task {name} with {cc.num_jobs} jobs, for LC fit {fit_name} on simulation {sim_name} and index {i}"
                        )
                        num_gen += 1
                        tasks.append(cc)

            if num_gen == 0:
                Task.fail_config(f"Classifier {clas_name} with masks |{mask}|{mask_sim}|{mask_fit}| matched no combination of sims and fits")
        return tasks
