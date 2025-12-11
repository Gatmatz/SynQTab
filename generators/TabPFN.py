from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, TabPFNUnsupervisedModel
from configs import TabPFNSettings
from generators.Generator import Generator
import torch

class TabPFN(Generator):
    """
    TabPFN synthetic data generator using TabPFN Unsupervised Model.
    """
    def __init__(self, settings: TabPFNSettings):
        super().__init__()
        self.settings = settings.to_dict()
        self.generator = None

    def generate(self, initial_df, categorical_features: list = None) -> torch.Tensor:
        """
        Generate synthetic data using TabPFN Unsupervised Model.
        :param initial_df: The initial dataframe to base synthetic data generation on.
        :param categorical_features: the list of indices for categorical features.
        :return:
        """
        tabpfn_clf = TabPFNClassifier(categorical_features_indices = categorical_features) # Model to handle categorical features
        tabpfn_reg = TabPFNRegressor(categorical_features_indices = categorical_features)  # Model to handle numerical features
        model = TabPFNUnsupervisedModel(tabpfn_clf, tabpfn_reg) # Unsupervised model for synthetic data generation

        # Manually set categorical features, don't let it infer
        if categorical_features is not None:
            model.set_categorical_features(categorical_features) # This does not work

        model.fit(initial_df)

        # Generate synthetic data using
        synthetic_X = model.generate_synthetic_data(
            self.settings["n_samples"],
            self.settings["temperature"],
            self.settings["n_permutations"]
        )

        return synthetic_X
