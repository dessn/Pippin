from abc import abstractmethod

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
        pass

    @abstractmethod
    def train(self, force_refresh):
        pass

    @staticmethod
    def get_requirements(config):
        return True, True

    def get_fit_dependency(self):
        for t in self.dependencies:
            if isinstance(t, SNANALightCurveFit):
                return t.output
        return None

    def get_simulation_dependency(self):
        for t in self.dependencies:
            if isinstance(t, SNANASimulation):
                return t.output
        return None

    def _run(self, force_refresh):
        if self.mode == Classifier.TRAIN:
            return self.train(force_refresh)
        elif self.mode == Classifier.PREDICT:
            return self.predict(force_refresh)

    def get_prob_column_name(self):
        return f"PROB_{self.name}"
