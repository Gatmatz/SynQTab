from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Self, Tuple
import pandas as pd

from synqtab.data.Dataset import Dataset
from synqtab.enums.data import DataPerfectness
from synqtab.enums.evaluators import EvaluationMethod
from synqtab.enums.generators import GeneratorModel
from synqtab.generators.Generator import Generator
from synqtab.errors.DataError import DataError
from synqtab.errors.DataErrorApplicability import DataErrorApplicability
from synqtab.reproducibility.ReproducibleOperations import (
    ReproducibilityError, ReproducibleOperations
)


class Experiment(ABC):

    def __init__(
        self,
        dataset: Dataset,
        generator: GeneratorModel,
        data_error: Optional[DataError] = None,
        data_perfectness: DataPerfectness = DataPerfectness.PERFECT,
        evaluators: Optional[list[EvaluationMethod]] = None,
        options: Optional[Dict[Any, Any]] = None,
    ):
        self.dataset = dataset
        self.generator = generator
        self.data_error = data_error
        self.data_perfectness = data_perfectness
        self.evaluators = evaluators
        self.options = options
        
        self.initialize_other_attributes()
        
    def run(self) -> Self:
        return self
    
    def persist(self) -> Self:
        return self
    
    def populate_tasks(self) -> Self:
        return self
        

    @abstractmethod
    def data_error_applicability(self) -> DataErrorApplicability:
        pass

    def find_numerical_categorical_columns(self) -> None:
        self.categorical_columns = [
            column
            for column in self.corrupted_data.columns
            if isinstance(self.corrupted_data[column].dtype, pd.CategoricalDtype)
        ]
        self.numeric_columns = [
            column
            for column in self.corrupted_data.columns
            if pd.api.types.is_numeric_dtype(self.corrupted_data[column].dtype)
        ]

    def identify_rows_to_corrupt(self, data: pd.DataFrame) -> None:
        self.rows_to_corrupt = ReproducibleOperations.sample_from(
            elements=data.index.to_list(), how_many=self.row_fraction * data.shape[0]
        )

    def identify_columns_to_corrupt(self) -> None:
        total_number_of_columns = len(self.numeric_columns + self.categorical_columns)
        number_of_columns_to_corrupt = self.column_fraction * total_number_of_columns

        match self.dataErrorApplicability():
            case DataErrorApplicability.CATEGORICAL_ONLY:
                self.columns_to_corrupt = ReproducibleOperations.sample_from(
                    elements=self.categorical_columns,
                    how_many=number_of_columns_to_corrupt,
                )

            case DataErrorApplicability.NUMERIC_ONLY:
                self.columns_to_corrupt = ReproducibleOperations.sample_from(
                    elements=self.numeric_columns,
                    how_many=number_of_columns_to_corrupt
                )

            case DataErrorApplicability.ANY_COLUMN:
                self.columns_to_corrupt = ReproducibleOperations.sample_from(
                    elements=self.numeric_columns + self.categorical_columns,
                    how_many=number_of_columns_to_corrupt,
                )

            case _ as not_implemented_category:
                raise NotImplementedError(
                    f"Unknown data error applicability type. Got {not_implemented_category}. \
                        Valid options: {[option.value for option in DataErrorApplicability]}."
                )

    def corrupt(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List, List]:
        # prepare for corruption
        self.corrupted_data = data.copy(deep=True)
        self.find_numerical_categorical_columns()
        self.identify_rows_to_corrupt(data)
        self.identify_columns_to_corrupt()

        # apply corruption; this is meant to be overriden for each specific data error type
        self.corrupted_data = self._apply_corruption(
            data_to_corrupt=self.corrupted_data,
            rows_to_corrupt=self.rows_to_corrupt,
            columns_to_corrupt=self.columns_to_corrupt,
        )

        # return tuple in a standardized way; single-point of change if needed
        return self.corruption_result_output_tuple()

    @abstractmethod
    def _apply_corruption(
        self,
        data_to_corrupt: pd.DataFrame,
        rows_to_corrupt: List,
        columns_to_corrupt: List,
    ) -> pd.DataFrame:
        """Apply the data corruption logic on the data to corrupt. The rows and columns
        to be corrupted are already provided as arguments. This function is meant to only
        apply the actual data corruption on the provided `data_to_corrupt` and return it.

        Args:
            data_to_corrupt (pd.DataFrame): The data to apply corruption on. You should modify it and return it.
            rows_to_corrupt (List): The rows to apply corruption on.
            columns_to_corrupt (List): The columns to apply corruption on. Data error compatibility is already ensured.

        Returns:
            pd.DataFrame: The corrupted data frame.
        """
        pass

    def corruption_result_output_tuple(self) -> Tuple[pd.DataFrame, List, List]:
        return self.corrupted_data, self.rows_to_corrupt, self.columns_to_corrupt
