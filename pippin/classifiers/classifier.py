from abc import abstractmethod, ABC

import logging


class Classifier(ABC):
    def __init__(self, light_curve_dir, fit_dir, output_dir, options):
        self.light_curve_dir = light_curve_dir
        self.fit_dir = fit_dir
        self.output_dir = output_dir
        self.option = options
        self.logger = logging.getLogger("pippin")

    @abstractmethod
    def classify(self):
        pass

