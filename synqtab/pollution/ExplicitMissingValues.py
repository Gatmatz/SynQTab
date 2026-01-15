import numpy as np

from synqtab.pollution import DataError
from synqtab.pollution.DataErrorApplicability import DataErrorApplicability


class ExplicitMissingValues(DataError):

    MISSING_VALUE = np.nan

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.ANY_COLUMN

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        data_to_corrupt.loc[rows_to_corrupt, columns_to_corrupt] = self.MISSING_VALUE
        return data_to_corrupt
