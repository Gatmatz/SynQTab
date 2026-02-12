from synqtab.generators import Generator

class HMASynthesizer(Generator):
    def __init__(self):
        super().__init__()
        self.generator = None

    def generate(self, data:dict, metadata, scale: float):
        from synqtab.reproducibility import ReproducibleOperations
        
        self.generator = ReproducibleOperations.get_hma_synthesizer_model(metadata)

        # Ensure reproducibility before fit
        ReproducibleOperations._ensure_reproducibility()
        self.generator.fit(data)

        # Ensure reproducibility before sample
        ReproducibleOperations._ensure_reproducibility()
        samples = self.generator.sample(scale=scale)
        return samples