from pollution import DataError
from pollution.DataErrorApplicability import DataErrorApplicability
from reproducibility import ReproducibleOperations


# based on https://github.com/schelterlabs/jenga/blob/a8bd74a588176e64183432a0124553c774adb20d/src/jenga/corruptions/numerical.py#L29
class Outliers(DataError):

    SCALE_FACTORS = [10, 100, 1000]  # TODO: Discuss internally

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.NUMERIC_ONLY

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        for column_to_corrupt in columns_to_corrupt:
            scale_factor = ReproducibleOperations.sample_from(self.SCALE_FACTORS)
            data_to_corrupt.loc[rows_to_corrupt, column_to_corrupt] *= scale_factor

        return data_to_corrupt
