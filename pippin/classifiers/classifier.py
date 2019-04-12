from abc import abstractmethod, ABC

import logging

from pippin.task import Task


class Classifier(Task):
    TRAIN = 0
    PREDICT = 1

    def __init__(self, name, light_curve_dir, fit_dir, output_dir, mode, options):
        super().__init__(name, output_dir)
        self.light_curve_dir = light_curve_dir
        self.fit_dir = fit_dir
        self.options = options
        self.mode = mode

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @staticmethod
    def get_requirements(config):
        return True, True

    def run(self):
        if self.mode == Classifier.TRAIN:
            return self.train()
        elif self.mode == Classifier.PREDICT:
            return self.predict()
