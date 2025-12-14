from numpy import std

from pollution import DataError
from pollution.DataErrorApplicability import DataErrorApplicability
from reproducibility import ReproducibleOperations


# Based on https://github.com/schelterlabs/jenga/blob/a8bd74a588176e64183432a0124553c774adb20d/src/jenga/corruptions/numerical.py#L9
class GaussianNoise(DataError):

    # TODO: Discuss Internally
    SCALING_MIN = 1
    SCALING_MAX = 5
    NORMAL_MEAN = 0

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.NUMERIC_ONLY

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        for column_to_corrupt in columns_to_corrupt:
            stddev = std(data_to_corrupt[column_to_corrupt])
            scale = ReproducibleOperations.uniform(
                low=self.SCALING_MIN, high=self.SCALING_MAX
            )
            noise = ReproducibleOperations.normal(
                loc=self.NORMAL_MEAN, scale=scale * stddev, size=len(rows_to_corrupt)
            )
            data_to_corrupt.loc[rows_to_corrupt, column_to_corrupt] += noise

        return data_to_corrupt
