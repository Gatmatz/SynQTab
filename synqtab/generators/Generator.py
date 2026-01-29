from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

class Generator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, X_initial: pd.DataFrame, y_initial: pd.DataFrame, n_samples: int, metadata: dict[str, Any]) -> pd.DataFrame:
        """Train the generator model."""
        pass