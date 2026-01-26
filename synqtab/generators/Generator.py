from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

class Generator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, X_initial, y_initial, params: Optional[dict]=None) -> pd.DataFrame:
        """Train the generator model."""
        pass