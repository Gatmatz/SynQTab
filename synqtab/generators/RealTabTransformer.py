from typing import Any

import pandas as pd

from synqtab.generators import Generator


# https://colab.research.google.com/github/avsolatorio/RealTabFormer/blob/main/colab/REaLTabFormer_GeoValidator_Example.ipynb
class RealTabTransformer(Generator):
    def __init__(self):
        super().__init__()
        self.generator = None

    def generate(self, X_initial: pd.DataFrame, y_initial: pd.DataFrame, n_samples: int, metadata: dict[str, Any]):
        from synqtab.reproducibility import ReproducibleOperations
        
        self.generator = ReproducibleOperations.get_realtabformer_model(model_type="tabular")

        self.generator.fit(pd.concat([X_initial, y_initial], axis=1))
        samples = self.generator.sample(n_samples=n_samples)
        return samples
