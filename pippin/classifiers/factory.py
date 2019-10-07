from pippin.classifiers.perfect import PerfectClassifier
from pippin.classifiers.snirf import SnirfClassifier
from pippin.classifiers.nearest_neighbor import NearestNeighborClassifier
from pippin.classifiers.supernnova import SuperNNovaClassifier
from pippin.classifiers.fitprob import FitProbClassifier
from pippin.classifiers.unity import UnityClassifier


class ClassifierFactory:
    ids = {}

    @classmethod
    def get(cls, name):
        return cls.ids[name]

    @classmethod
    def add_factory(cls, classifier_class):
        cls.ids[classifier_class.__name__] = classifier_class


ClassifierFactory.add_factory(FitProbClassifier)
ClassifierFactory.add_factory(SuperNNovaClassifier)
ClassifierFactory.add_factory(SnirfClassifier)
ClassifierFactory.add_factory(NearestNeighborClassifier)
ClassifierFactory.add_factory(UnityClassifier)
ClassifierFactory.add_factory(PerfectClassifier)
