from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, TabPFNUnsupervisedModel
from synqtab.configs import TabPFNSettings
from synqtab.generators.Generator import Generator
import pandas as pd
import torch

class TabPFN(Generator):
    """
    TabPFN synthetic data generator using TabPFN Unsupervised Model.
    """
    def __init__(self, settings: TabPFNSettings):
        super().__init__()
        self.settings = settings.to_dict()
        self.generator = None

    def generate(self, X_initial, y_initial, task):
        # TODO extract categorical extraction to another generic utility class
        # TODO revise the generators to also get the random seed and pass it to classifier and regressor
        classifier = TabPFNClassifier()
        regressor = TabPFNRegressor()
        self.generator = TabPFNUnsupervisedModel(classifier, regressor)
        df = pd.concat([X_initial, y_initial], axis=1)
        df_tensor = torch.tensor(pd.get_dummies(df), dtype=torch.float32)
        self.generator.fit(df_tensor)
        synthetic_data = self.generator.generate_synthetic_data(
            n_samples=self.settings['n_samples'],
            t=self.settings.get('temperature', 1),
            n_permutations=self.settings.get('n_permutations', 3)
        )
        # TODO see how we can restore the one hot encoding for the quality evaluation (column-wise will not work otherwise)
        return pd.DataFrame(synthetic_data.detach().numpy())
