import pandas as pd

from synqtab.generators.Generator import Generator
from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations

# https://colab.research.google.com/github/avsolatorio/RealTabFormer/blob/main/colab/REaLTabFormer_GeoValidator_Example.ipynb


class RealTabTransformer(Generator):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.generator = None

    def generate(self, X_initial, y_initial, task: str):
        
        self.generator = ReproducibleOperations.get_realtabformer_model(
            model_type="tabular",
            gradient_accumulation_steps=self.settings.get(
                "gradient_accumulation_steps", 4
            ),
            logging_steps=self.settings.get("logging_steps", 100),
        )

        self.generator.fit(pd.concat([X_initial, y_initial], axis=1))
        samples = self.generator.sample(n_samples=self.settings["n_samples"])
        return samples
