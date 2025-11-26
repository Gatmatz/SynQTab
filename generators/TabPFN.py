from generators.Generator import Generator, Task
from tabpfgen import TabPFGen

class TabPFN(Generator):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.generator = None

    def train_model(self, X_initial, y_initial, task: str):
        self.generator = TabPFGen(n_sgld_steps=self.settings['n_sgld_steps'])
        if task == Task.CLASSIFICATION:
            X_synth, y_synth = self.generator.generate_classification(
                X_initial,
                y_initial,
                task,
                self.settings["n_classes"],
                self.settings["balance_classes"]
            )
        elif task == Task.REGRESSION:
            X_synth, y_synth = self.generator.generate_regression(
                X_initial,
                y_initial,
                task,
                self.settings["n_samples"],
                self.settings["use_quantiles"]
            )

        return X_synth, y_synth