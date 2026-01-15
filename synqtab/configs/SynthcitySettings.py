# TODO: I know we want something quick and dirty for now, 
# but for later on we could consider unifying the Settings instead of
# having one for TabPFN and another for Synthcity

from enum import Enum

class SynthcityModelOption(Enum):
    # generic models
    CTGAN = 'ctgan'
    NFLOW = 'nflow'
    RTVAE = 'rtvae'
    TVAE = 'tvae'
    DDPM = 'ddpm'
    ARF = 'arf'
    MARGINAL_DISTRIBUTIONS = 'marginal_distributions'
    BAYESIAN_NETWORK = 'bayesian_network'
    GREAT = 'great'
    
    # privacy-focused
    ADSGAN = 'adsgan'
    PATEGAN = 'pategan'
    AIM = 'aim'
    DPGAN = 'dpgan'
    DECAF = 'decaf'
    PRIVBAYES = 'privbayes'

class SynthcitySettings:
    def __init__(
        self, 
        model_name: SynthcityModelOption,
        n_samples: int = 1000,
    ):
        self.model_name = model_name.value
        self.n_samples = n_samples

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, settings_dict: dict):
        return cls(**settings_dict)
