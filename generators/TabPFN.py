from configs import TabPFNSettings
from generators.Generator import Generator
from tabpfgen import TabPFGen


class TabPFN(Generator):
    """
    TabPFN synthetic data generator.
    """
    def __init__(self, settings: TabPFNSettings):
        super().__init__()
        self.settings = settings.to_dict()
        self.generator = None

    def generate(self, X_initial, y_initial, task: str):
        """
        Generate synthetic data using TabPFN.
        :param X_initial: Feature matrix of the initial dataset.
        :param y_initial: Labels of the initial dataset.
        :param task: ML task type (classification or regression).
        :return:
        """
        self.generator = TabPFGen(n_sgld_steps=self.settings['n_sgld_steps'])
        X_initial_np = X_initial.to_numpy()
        if task == 'classification':
            X_synth, y_synth = self.generator.generate_classification(
                X_initial_np,
                y_initial,
                self.settings["n_samples"],
                self.settings["balance_classes"]
            )
        elif task == 'regression':
            X_synth, y_synth = self.generator.generate_regression(
                X_initial_np,
                y_initial,
                self.settings["n_samples"],
                self.settings["use_quantiles"]
            )

        return X_synth, y_synth
