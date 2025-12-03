import abc
from abc import ABC
from enum import Enum

class Generator(ABC):
    def __init__(self):
        super().__init__()

    def generate(self, X_initial, y_initial, task: str):
        """Train the generator model."""
        pass