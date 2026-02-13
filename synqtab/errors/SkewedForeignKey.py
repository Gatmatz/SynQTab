from synqtab.errors.DataError import DataError


class SkewedForeignKey(DataError):
    """
    Data error that creates a skewed foreign key distribution by making a large 
    percentage of rows have the same FK value.
    
    If row_fraction is 0.1, then 90% (1 - row_fraction) of the rows will have 
    the same FK value, while 10% remain with their original values.
    """

    def data_error_applicability(self):
        from synqtab.errors import DataErrorApplicability
        
        return DataErrorApplicability.ANY_COLUMN
    
    def full_name(self) -> str:
        return "Skewed Foreign Key"
    
    def short_name(self) -> str:
        from synqtab.enums import DataErrorType
        
        return str(DataErrorType.SKEWED_FK)

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt, **kwargs):
        from synqtab.reproducibility import ReproducibleOperations
        import pandas as pd
        
        for column_to_corrupt in columns_to_corrupt:
            # Get all non-null values from the column
            non_null_values = data_to_corrupt[column_to_corrupt].dropna()
            
            if len(non_null_values) == 0:
                continue
            
            # Pick the most common value, or a random one if all are equally common
            value_counts = non_null_values.value_counts()
            common_value = value_counts.index[0]  # Most frequent value
            
            # Calculate which rows should have the same value
            # row_fraction determines what stays different, so (1 - row_fraction) should be the same
            total_rows = len(data_to_corrupt)
            num_rows_to_skew = int((1 - self.row_fraction) * total_rows)
            
            # Sample rows to make the same (this is the majority)
            rows_to_make_same = ReproducibleOperations.sample_from(
                elements=data_to_corrupt.index.to_list(), 
                how_many=max(num_rows_to_skew, 1)
            )
            
            # Set these rows to have the common value
            data_to_corrupt.loc[rows_to_make_same, column_to_corrupt] = common_value

        return data_to_corrupt
