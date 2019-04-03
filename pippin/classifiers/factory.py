from pippin.classifiers.argonne import ArgonneClassifier
from pippin.classifiers.supernnova import SuperNNovaClassifier
from pippin.classifiers.toy import ToyClassifier


class ClassifierFactory:
    ids = {}

    @classmethod
    def get(cls, name):
        return cls.ids[name]

    @classmethod
    def add_factory(cls, classifierClass):
        cls.ids[classifierClass.__name__] = classifierClass

ClassifierFactory.add_factory(ToyClassifier)
ClassifierFactory.add_factory(SuperNNovaClassifier)
ClassifierFactory.add_factory(ArgonneClassifier)

if __name__ == "__main__":
    c = ClassifierFactory.get("ToyClassifier")
    print(c)