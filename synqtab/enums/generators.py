from synqtab.enums.EasilyStringifyableEnum import EasilyStringifyableEnum

class GeneratorModel(EasilyStringifyableEnum):
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
    REALTABFORMER = 'realtabformer'
    TABPFN = 'tabpfn'
    TABEBM = 'tabebm'
    
    # privacy-focused
    ADSGAN = 'adsgan'
    PATEGAN = 'pategan'
    AIM = 'aim'
    DPGAN = 'dpgan'
    DECAF = 'decaf'
    PRIVBAYES = 'privbayes'
    

GENERIC_MODELS = [
    GeneratorModel.CTGAN,
    GeneratorModel.NFLOW,
    GeneratorModel.RTVAE,
    GeneratorModel.TVAE,
    GeneratorModel.DDPM,
    GeneratorModel.ARF,
    GeneratorModel.MARGINAL_DISTRIBUTIONS,
    GeneratorModel.BAYESIAN_NETWORK,
    GeneratorModel.GREAT,
    GeneratorModel.REALTABFORMER,
    GeneratorModel.TABPFN,
    GeneratorModel.TABEBM,
]


PRIVACY_MODELS = [
    GeneratorModel.ADSGAN,
    GeneratorModel.PATEGAN,
    GeneratorModel.AIM,
    GeneratorModel.DPGAN,
    GeneratorModel.DECAF,
    GeneratorModel.PRIVBAYES,
]

class SynthcityModelOption(EasilyStringifyableEnum):
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
    
SYNTHCITY_GENERIC_MODELS = [
    SynthcityModelOption.CTGAN,
    SynthcityModelOption.NFLOW,
    SynthcityModelOption.RTVAE,
    SynthcityModelOption.TVAE,
    SynthcityModelOption.DDPM,
    SynthcityModelOption.ARF,
    SynthcityModelOption.MARGINAL_DISTRIBUTIONS,
    SynthcityModelOption.BAYESIAN_NETWORK,
]

SYNTHCITY_PRIVACY_MODELS = [
    SynthcityModelOption.ADSGAN,
    SynthcityModelOption.PATEGAN,
    SynthcityModelOption.AIM,
    SynthcityModelOption.DPGAN,
    SynthcityModelOption.DECAF,
    SynthcityModelOption.PRIVBAYES,
]

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

class TabPFNSettings:
    def __init__(self, n_samples: int = 100, temperature: float = 1.0, n_permutations: int = 3):
        self.n_samples = n_samples
        self.temperature = temperature
        self.n_permutations = n_permutations

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, settings_dict: dict):
        return cls(**settings_dict)
