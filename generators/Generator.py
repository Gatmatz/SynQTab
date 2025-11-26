import abc
from abc import ABC
from enum import Enum

class Task(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class Generator(ABC):
    def __init__(self):
        super().__init__()

    def train_model(self, X_initial, y_initial, task: str):
        """Train the generator model."""
        pass