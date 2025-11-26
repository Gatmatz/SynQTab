from generators.Generator import Generator, Task
from realtabformer import REaLTabFormer

class RealTabTransformer(Generator):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.generator = None

    def train_model(self, data, task: str):
        self.generator = REaLTabFormer(
            model_type='tabular',
            gradient_accumulation_steps=self.settings['gradient_accumulation_steps'],
            logging_steps=self.settings['logging_steps']
        )

        self.generator.fit(data)

        samples = self.generator.sample(n_samples=self.settings['n_samples'])

        return samples