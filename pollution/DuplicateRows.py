import pandas as pd

from pollution import DataError
from pollution.DataErrorApplicability import DataErrorApplicability
from reproducibility import ReproducibleOperations


class DuplicateRows(DataError):

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.ANY_COLUMN # columns are not actually used for this error type

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        data_to_corrupt = pd.concat([data_to_corrupt, data_to_corrupt.loc[rows_to_corrupt]])
        data_to_corrupt = ReproducibleOperations.shuffle_reindex_dataframe(data_to_corrupt)
        return data_to_corrupt
