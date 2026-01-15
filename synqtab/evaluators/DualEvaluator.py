from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

class DualEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    """

    @abstractmethod
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict[str, Any], holdout_table:Optional[pd.DataFrame]) -> Any:
        """
        Evaluates the quality of synthetic data against real data.

        Args:
            real_data (pd.DataFrame): The original real dataset.
            synthetic_data (pd.DataFrame): The generated synthetic dataset.
            metadata (Dict[str, Any], optional): Metadata describing the dataset structure (e.g., SDMetrics metadata).
            holdout_table: Optional[pd.DataFrame]: An optional holdout dataset for evaluation.

        """
        pass