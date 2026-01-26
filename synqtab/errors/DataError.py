from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from synqtab.errors.DataErrorApplicability import DataErrorApplicability
from synqtab.reproducibility import ReproducibleOperations


class DataError(ABC):

    def __init__(
        self,
        # random_seed: int | float,
        row_fraction: float,
        column_fraction: float = 0.2,
        options: Optional[Dict[Any, Any]] = None,
    ):
        # self.random_seed = random_seed
        self.row_fraction = row_fraction
        self.column_fraction = column_fraction
        self.options = options
        self.validate_instance_attributes()
        self.initialize_other_attributes()

    @abstractmethod
    def data_error_applicability(self) -> DataErrorApplicability:
        pass
    
    @abstractmethod
    def full_name(self) -> str:
        pass
    
    @abstractmethod
    def short_name(self) -> str:
        pass

    def validate_instance_attributes(self) -> None:
        # self.validate_random_seed()
        self.validate_row_fraction()
        self.validate_column_fraction()
        self.validate_options()

    def initialize_other_attributes(self):
        # ReproducibleOperations.set_random_seed(self.random_seed)
        self.categorical_columns = []
        self.numeric_columns = []
        self.rows_to_corrupt = []
        self.columns_to_corrupt = []
        self.corrupted_data = None

    # def validate_random_seed(self) -> None:
    #     if not self.random_seed:
    #         raise ReproducibilityError(
    #             f"Setting the random seed to None is not reproducible. I was expecting any integer or float number."
    #         )

    def validate_row_fraction(self) -> None:
        if not 0 <= self.row_fraction <= 1:
            raise ValueError(
                f"Row fraction must be in the range [0, 1]. Got {self.row_fraction}."
            )

    def validate_column_fraction(self) -> None:
        if not 0 <= self.column_fraction <= 1:
            raise ValueError(
                f"Column fraction must be in the range [0, 1]. Got {self.column_fraction}."
            )

    def validate_options(self) -> None:
        pass

    def find_numerical_categorical_columns(self, **kwargs) -> None:
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

    def identify_rows_to_corrupt(self, data: pd.DataFrame, **kwargs) -> None:
        self.rows_to_corrupt = ReproducibleOperations.sample_from(
            elements=data.index.to_list(), how_many=int(max(self.row_fraction * data.shape[0], 1))
        )

    def identify_columns_to_corrupt(self, **kwargs) -> None:
        total_number_of_columns = len(self.numeric_columns + self.categorical_columns)
        number_of_columns_to_corrupt = int(max(self.column_fraction * total_number_of_columns, 1))

        match self.data_error_applicability():
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
                    f"Unknown data error applicability type. Got {not_implemented_category}. " +
                    f"Valid options: {[option.value for option in DataErrorApplicability]}."
                )

    def corrupt(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List, List]:
        # prepare for corruption
        self.corrupted_data = data.copy(deep=True)
        self.find_numerical_categorical_columns(**kwargs)
        self.identify_rows_to_corrupt(data, **kwargs)
        self.identify_columns_to_corrupt(**kwargs)
        
        # columns_to_corrupt == [] can happen if e.g., a categorical error must be applied but no categorical cols exist
        if self.columns_to_corrupt: 
            # apply corruption; this is meant to be overriden for each specific data error type
            self.corrupted_data = self._apply_corruption(
                data_to_corrupt=self.corrupted_data,
                rows_to_corrupt=self.rows_to_corrupt,
                columns_to_corrupt=self.columns_to_corrupt,
                **kwargs,
            )

        # return tuple in a standardized way; single-point of change if needed
        return self.corruption_result_output_tuple(**kwargs)

    @abstractmethod
    def _apply_corruption(
        self,
        data_to_corrupt: pd.DataFrame,
        rows_to_corrupt: List,
        columns_to_corrupt: List,
        **kwargs,
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

    def corruption_result_output_tuple(self, **kwargs) -> Tuple[pd.DataFrame, List, List]:
        return self.corrupted_data, self.rows_to_corrupt, self.columns_to_corrupt
