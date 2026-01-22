from abc import ABC
from typing import Optional

class Generator(ABC):
    def __init__(self):
        super().__init__()

    def generate(self, X_initial, y_initial, params: Optional[dict]=None):
        """Train the generator model."""
        pass