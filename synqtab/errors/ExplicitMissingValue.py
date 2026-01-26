import numpy as np

from synqtab.errors.DataError import DataError


class ExplicitMissingValue(DataError):

    MISSING_VALUE = np.nan

    def data_error_applicability(self):
        from synqtab.errors import DataErrorApplicability
        
        return DataErrorApplicability.ANY_COLUMN
    
    def full_name(self):
        return "Explicit Missing Values"
    
    def short_name(self):
        return "EMV"

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt, **kwargs):
        data_to_corrupt.loc[rows_to_corrupt, columns_to_corrupt] = self.MISSING_VALUE
        return data_to_corrupt
