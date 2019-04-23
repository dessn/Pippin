from pippin.classifiers.argonne import ArgonneClassifier
from pippin.classifiers.nearest_neighbor import NearestNeighborClassifier
from pippin.classifiers.supernnova import SuperNNovaClassifier
from pippin.classifiers.fitprob import FitProbClassifier


class ClassifierFactory:
    ids = {}

    @classmethod
    def get(cls, name):
        return cls.ids[name]

    @classmethod
    def add_factory(cls, classifierClass):
        cls.ids[classifierClass.__name__] = classifierClass

ClassifierFactory.add_factory(FitProbClassifier)
ClassifierFactory.add_factory(SuperNNovaClassifier)
ClassifierFactory.add_factory(ArgonneClassifier)
ClassifierFactory.add_factory(NearestNeighborClassifier)

if __name__ == "__main__":
    c = ClassifierFactory.get("ToyClassifier")
    print(c)