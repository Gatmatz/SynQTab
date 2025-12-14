from pollution import DataError
from pollution.DataErrorApplicability import DataErrorApplicability


class TyposSwappedLetters(DataError):

    def data_error_applicability(self) -> DataErrorApplicability:
        return DataErrorApplicability.CATEGORICAL_ONLY

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt):
        # TODO
        return data_to_corrupt
