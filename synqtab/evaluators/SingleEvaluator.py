from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

class SingleEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    """

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> Any:
        """
        Evaluates the quality of synthetic data against real data.

        Args:
            real_data (pd.DataFrame): The original real dataset.
            synthetic_data (pd.DataFrame): The generated synthetic dataset.
            metadata (Dict[str, Any], optional): Metadata describing the dataset structure (e.g., SDMetrics metadata).

        Returns:
            Any: The result of the evaluation.
        """
        pass