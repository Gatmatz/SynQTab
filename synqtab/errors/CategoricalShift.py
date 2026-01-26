from synqtab.errors.DataError import DataError


class CategoricalShift(DataError):

    def data_error_applicability(self):
        from synqtab.errors import DataErrorApplicability
        
        return DataErrorApplicability.CATEGORICAL_ONLY
    
    def full_name(self) -> str:
        return "Categorical Shift"
    
    def short_name(self) -> str:
        from synqtab.enums import DataErrorType
        
        return str(DataErrorType.CATEGORICAL_SHIFT)

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt, **kwargs):
        from synqtab.reproducibility import ReproducibleOperations
        
        for column_to_corrupt in columns_to_corrupt:
            distinct_values = data_to_corrupt[column_to_corrupt].value_counts().index
            permuted_distinct_values = ReproducibleOperations.permutation(distinct_values)
            replacement_values = data_to_corrupt.loc[rows_to_corrupt, column_to_corrupt] \
                                .replace(distinct_values, permuted_distinct_values)
            data_to_corrupt.loc[rows_to_corrupt, column_to_corrupt] = replacement_values

        return data_to_corrupt
