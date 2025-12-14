from pollution import DataError
import pandas as pd
from typing import List, Tuple

from pollution.DataErrorApplicability import DataErrorApplicability


class ClassImbalance(DataError):

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.ANY_COLUMN

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        # TODO
        return data_to_corrupt
