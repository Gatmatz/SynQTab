from synqtab.errors.DataError import DataError


class OrphanedForeignKey(DataError):
    """
    Data error that creates orphaned foreign key references by replacing FK values
    with random values that don't exist in the column, simulating referential integrity violations.
    """

    def data_error_applicability(self):
        from synqtab.errors import DataErrorApplicability
        
        return DataErrorApplicability.ANY_COLUMN
    
    def full_name(self) -> str:
        return "Orphaned Foreign Key"
    
    def short_name(self) -> str:
        from synqtab.enums import DataErrorType
        
        return str(DataErrorType.ORPHANED_FK)

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt, **kwargs):
        import pandas as pd
        
        for column_to_corrupt in columns_to_corrupt:
            # Set the corrupted rows to null values
            data_to_corrupt.loc[rows_to_corrupt, column_to_corrupt] = pd.NA

        return data_to_corrupt
