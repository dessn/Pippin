from abc import abstractmethod, ABC

import logging

from pippin.task import Task


class Classifier(Task):
    def __init__(self, name, light_curve_dir, fit_dir, output_dir, options):
        super().__init__(name, output_dir)
        self.light_curve_dir = light_curve_dir
        self.fit_dir = fit_dir
        self.options = options

    @abstractmethod
    def classify(self):
        pass

    @staticmethod
    def get_requirements(config):
        return True, True

    def run(self):
        self.classify()
