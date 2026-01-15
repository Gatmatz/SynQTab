from synqtab.pollution import DataError

from synqtab.pollution.DataErrorApplicability import DataErrorApplicability
from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations


class CategoricalShift(DataError):

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.CATEGORICAL_ONLY

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        for column_to_corrupt in columns_to_corrupt:
            distinct_values = data_to_corrupt[column_to_corrupt].value_counts().index
            permuted_distinct_values = ReproducibleOperations.permutation(distinct_values)
            replacement_values = data_to_corrupt.loc[rows_to_corrupt, column_to_corrupt] \
                                .replace(distinct_values, permuted_distinct_values)
            data_to_corrupt.loc[rows_to_corrupt, column_to_corrupt] = replacement_values

        return data_to_corrupt
