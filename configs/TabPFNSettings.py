class TabPFNSettings:
    def __init__(self, n_sgld_steps: int = 10, n_samples: int = 1, balance_classes: bool = False):
        self.n_sgld_steps = n_sgld_steps
        self.n_samples = n_samples
        self.balance_classes = balance_classes

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, settings_dict: dict):
        return cls(**settings_dict)
