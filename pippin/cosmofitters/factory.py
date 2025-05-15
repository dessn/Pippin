from pippin.cosmofitters.wfit import WFit
from pippin.cosmofitters.cosmomc import CosmoMC


class FitterFactory:
    ids = {}

    @classmethod
    def get(cls, name):
        return cls.ids.get(name)

    @classmethod
    def add_factory(cls, fitter_class) -> None:
        cls.ids[fitter_class.__name__.lower()] = fitter_class


FitterFactory.add_factory(CosmoMC)
FitterFactory.add_factory(WFit)
