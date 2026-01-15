class PollutionSettings:
    def __init__(self, type: str = 'MCAR', random_seed: int = 42, row_percent: int = 10, column_percent: int = 10):
        self.type = type
        self.random_seed = random_seed
        self.row_percent = row_percent
        self.column_percent = column_percent

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, settings_dict: dict):
        return cls(**settings_dict)
