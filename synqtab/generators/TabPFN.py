from typing import Optional

from synqtab.generators import Generator


class TabPFN(Generator):
    """
    TabPFN synthetic data generator using TabPFN Unsupervised Model.
    """
    def __init__(self):
        super().__init__()
        self.generator = None

    def generate(self, X_initial, y_initial, params: Optional[dict]=None):
        import torch
        import pandas as pd
        from tabpfn_extensions import TabPFNUnsupervisedModel
        from synqtab.reproducibility import ReproducibleOperations
        
        # TODO extract categorical extraction to another generic utility class
        # TODO revise the generators to also get the random seed and pass it to classifier and regressor
        classifier = ReproducibleOperations.get_tabpfn_classifier_model()
        regressor = ReproducibleOperations.get_tabpfn_regression_model()
        self.generator = TabPFNUnsupervisedModel(classifier, regressor)
        df = pd.concat([X_initial, y_initial], axis=1)
        df_tensor = torch.tensor(pd.get_dummies(df), dtype=torch.float32)
        self.generator.fit(df_tensor)
        synthetic_data = self.generator.generate_synthetic_data(
            n_samples=self.params['n_samples'],
            t=self.params.get('temperature', 1),
            n_permutations=self.params.get('n_permutations', 3)
        )
        # TODO see how we can restore the one hot encoding for the quality evaluation (column-wise will not work otherwise)
        return pd.DataFrame(synthetic_data.detach().numpy())
