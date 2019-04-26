from pippin.classifiers.classifier import Classifier
from pippin.task import Task


class NearestNeighborClassifier(Classifier):
    def __init__(self, name, output_dir, dependencies, mode, options):
        super().__init__(name, output_dir, dependencies, mode, options)
        self.logger.info(f"Creating Nearest Neighbor classifier with options: {options}")

    def predict(self):
        return False

    def train(self):
        return False

    def check_completion(self):
        self.output = {
            "name": self.name,
            "output_dir": self.output_dir
        }
        self.logger.critical("CRITICAL ERROR, this hasn't been implemented yet")
        return Task.FINISHED_FAILURE
