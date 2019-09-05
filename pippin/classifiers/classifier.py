from abc import abstractmethod

from pippin.dataprep import DataPrep
from pippin.task import Task
from pippin.snana_sim import SNANASimulation
from pippin.snana_fit import SNANALightCurveFit


class Classifier(Task):
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

    def get_prob_column_name(self):
        m = self.get_model_classifier()
        if m is None:
            name = f"PROB_{self.name}"
            use_sim, use_fit = self.get_requirements(self.options)
            if use_fit:
                name += "_" + self.get_fit_dependency()["name"] + "_" + self.get_fit_dependency()["sim_name"]
            else:
                name += "_" + self.get_simulation_dependency()["name"]
            return name
        else:
            return m.output["prob_column_name"]