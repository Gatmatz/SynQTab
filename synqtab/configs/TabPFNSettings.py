class TabPFNSettings:
    def __init__(self, n_samples: int = 100, temperature: float = 1.0, n_permutations: int = 3):
        self.n_samples = n_samples
        self.temperature = temperature
        self.n_permutations = n_permutations

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, settings_dict: dict):
        return cls(**settings_dict)
