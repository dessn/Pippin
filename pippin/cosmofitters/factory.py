from pippin.cosmofitters.cosmomc import CosmoMC
from pippin.cosmofitters.wfit import WFit

class FitterFactory:
    ids = {}

    @classmethod
    def get(cls, name):
        return cls.ids.get(name)

    @classmethod
    def add_factory(cls, fitter_class):
        cls.ids[fitter_class.__name__.lower()] = fitter_class

FitterFactory.add_factory(CosmoMC)
FitterFactory.add_factory(WFit)
