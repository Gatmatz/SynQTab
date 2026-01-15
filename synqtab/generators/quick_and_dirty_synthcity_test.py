# Make sure you have uv: https://docs.astral.sh/uv/guides/install-python/
# only for the first time: from the root directory of the project: `uv venv && uv pip install -r requirements.txt && cd generators && uv run quick_and_dirty_synthcity_test.py`
# for any next execution: from the `generators` directory: `uv run quick_and_dirty_synthcity_test.py`

# The script takes approximately 3mins in cobra

from enum import Enum
from sklearn.datasets import load_diabetes
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import pandas as pd


N_SAMPLES = 1000

# Source of truth for synthcity models; docs are misleading
generic_models=['uniform_sampler', 'ddpm', 'marginal_distributions', 'rtvae', 'tvae', 'arf', 'nflow', 'dummy_sampler', 'ctgan']
privacy_models=['adsgan', 'pategan', 'aim', 'dpgan']

# DUPLICATED FOR CONVENIENCE. This should not be here on actual implementation
class SynthcityModelOption(Enum):
    # generic models
    CTGAN = 'ctgan'
    NFLOW = 'nflow'
    RTVAE = 'rtvae'
    TVAE = 'tvae'
    DDPM = 'ddpm'
    ARF = 'arf'
    MARGINAL_DISTRIBUTIONS = 'marginal_distributions'
    
    # privacy-focused
    ADSGAN = 'adsgan'
    PATEGAN = 'pategan'
    AIM = 'aim'
    DPGAN = 'dpgan'

# DUPLICATED FOR CONVENIENCE. This should not be here on actual implementation
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
    

class SynthcityGenerator():
    """Hierarchy class for generators from the syntcity package
    https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial0_basic_examples.ipynb
    """
    
    def __init__(self, settings: SynthcitySettings):
        super().__init__()
        self.settings = settings.to_dict()
        self.generator = None
    
    def generate(self, X_initial, y_initial, task):
        loader = GenericDataLoader(
            pd.concat([X_initial, y_initial], axis=1),
            target_column=y_initial.columns[0]
        )
        self.generator = Plugins().get(self.settings["model_name"])
        self.generator.fit(loader)
        return self.generator.generate(count=self.settings["n_samples"]).dataframe()

def main():
    from time import time
    
    start = time()
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    for model in SynthcityModelOption:
        model_start = time()
        print("=" * 50)
        print(model.name)
        print("=" * 50)
        
        settings = SynthcitySettings(model_name=model, n_samples=N_SAMPLES)
        generator = SynthcityGenerator(settings=settings)
        generated_data = generator.generate(X, y.to_frame(), "")
        
        print(generated_data.describe())
        print(f"Time elapsed: {time() - model_start} sec")
    
    print()
    print()
    print(f"Total elapsed time: {time() - start} sec")
    
if __name__ == '__main__':
    main()