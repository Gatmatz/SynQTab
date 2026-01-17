from synqtab.pollution import DataError
from synqtab.pollution.DataErrorApplicability import DataErrorApplicability
from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations


class RepresentationalInconsistencies(DataError):

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.CATEGORICAL_ONLY

    def _apply_typo(self, categorical_value: str) -> str:
        """Applies a typo to a string value. Randomly (yet reproducibly)
        selects one of the following typo types:
        - extra letter -> see _apply_typo_extra_letter;
        - missing letter -> see _apply_typo_missing_letter;
        - swapped letter -> see _apply_typo_swapped_letter.

        Args:
            categorical_value (str): the value to apply typo on

        Returns:
            str: the value with the typo
        """
        return ReproducibleOperations.sample_from(
            elements=[
                self._apply_typo_extra_letter,
                self._apply_typo_missing_letter,
                self._apply_typo_swapped_letter,
            ],
            how_many=1,
        )[0](categorical_value)

    def _apply_typo_extra_letter(self, categorical_value: str) -> str:
        """Applies a typo to a string value by adding an extra letter
        right after its occurrence, e.g., pollution -> poolution.

        Args:
            categorical_value (str): the value to applly the typo on

        Returns:
            str: the value with the typo
        """
        extra_letter_index = ReproducibleOperations.sample_from(
            elements=range(len(categorical_value)), how_many=1
        )[0]
        return (
            categorical_value[:extra_letter_index]
            + 2 * categorical_value[extra_letter_index]
            + categorical_value[extra_letter_index + 1 :]
        )

    def _apply_typo_missing_letter(self, categorical_value: str) -> str:
        """Applies a typo to a string value by removing a letter,
        e.g., pollution -> polution.

        Args:
            categorical_value (str): the value to apply the typo on

        Returns:
            str: the value with the typo
        """
        missing_char_index = ReproducibleOperations.sample_from(
            elements=range(len(categorical_value)), how_many=1
        )[0]
        return (
            categorical_value[:missing_char_index]
            + categorical_value[missing_char_index + 1 :]
        )

    def _apply_typo_swapped_letter(self, categorical_value: str) -> str:
        """Applies a typo to a string value by swapping two neighboring
        letters, e.g., pollution -> pollutoin

        Args:
            categorical_value (str): the value to apply the typo on

        Returns:
            str: the value with the typo
        """
        left_swapped_char_index = ReproducibleOperations.sample_from(
            elements=range(len(categorical_value) - 1), how_many=1
        )[0]
        right_swapped_char_index = left_swapped_char_index + 1
        return (
            categorical_value[:left_swapped_char_index]
            + categorical_value[right_swapped_char_index]
            + categorical_value[left_swapped_char_index]
            + categorical_value[right_swapped_char_index + 1 :]
        )

    def _apply_corruption_to_categorical_column(
        self, data_to_corrupt, rows_to_corrupt, categorical_column_to_corrupt
    ):
        """Applies corruption (typos) to a categorical column.

        Args:
            data_to_corrupt (pd.DataFrame): the data to corrupt
            rows_to_corrupt (list): the rows to corrupt
            categorical_column_to_corrupt (str): the column name to corrupt

        Returns:
            pd.DataFrame: the `data_to_corrupt`, after applying the corurption on top of it
        """
        distinct_values = (
            data_to_corrupt[categorical_column_to_corrupt].value_counts().index
        )
        distinct_values_with_typos = [
            self._apply_typo(distinct_value) for distinct_value in distinct_values
        ]
        replacement_values = data_to_corrupt.loc[
            rows_to_corrupt, categorical_column_to_corrupt
        ].replace(distinct_values, distinct_values_with_typos)
        data_to_corrupt.loc[rows_to_corrupt, categorical_column_to_corrupt] = (
            replacement_values
        )
        return data_to_corrupt

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        for column_to_corrupt in columns_to_corrupt:
            data_to_corrupt = self._apply_corruption_to_categorical_column(
                data_to_corrupt, rows_to_corrupt, column_to_corrupt
            )

        return data_to_corrupt
