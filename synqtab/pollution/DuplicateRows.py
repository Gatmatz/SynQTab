import pandas as pd

from synqtab.pollution import DataError
from synqtab.pollution.DataErrorApplicability import DataErrorApplicability
from synqtab.reproducibility import ReproducibleOperations


class DuplicateRows(DataError):

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.ANY_COLUMN # columns are not actually used for this error type

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        data_to_corrupt = pd.concat([data_to_corrupt, data_to_corrupt.loc[rows_to_corrupt]])
        data_to_corrupt = ReproducibleOperations.shuffle_reindex_dataframe(data_to_corrupt)
        return data_to_corrupt
