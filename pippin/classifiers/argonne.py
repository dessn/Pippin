from pippin.classifiers.classifier import Classifier
from pippin.config import get_config
from pippin.task import Task


class ArgonneClassifier(Classifier):
    def __init__(self, name, output_dir, dependencies, mode, options):
        super().__init__(name, output_dir, dependencies, mode, options)
        self.global_config = get_config()

    def predict(self, force_refresh):
        pass

    def train(self, force_refresh):
        pass

    def _check_completion(self):
        return Task.FINISHED_FAILURE

    @staticmethod
    def get_requirements(options):
        return False, True
