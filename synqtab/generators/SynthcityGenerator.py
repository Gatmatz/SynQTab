import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from synqtab.generators.Generator import Generator
from synqtab.configs.SynthcitySettings import SynthcitySettings


class SynthcityGenerator(Generator):
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
