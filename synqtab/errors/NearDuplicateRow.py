from synqtab.errors.Inconsistency import Inconsistency


class NearDuplicateRow(Inconsistency):

    def data_error_applicability(self):
        from synqtab.errors import DataErrorApplicability
        
        return DataErrorApplicability.ANY_COLUMN
    
    def _apply_corruption_to_numeric_column(
        self, data_to_corrupt, rows_to_corrupt, numeric_column_to_corrupt, **kwargs
    ):
        """Applies corruption (implicit missing values) to a numeric column.

        Args:
            data_to_corrupt (pd.DataFrame): the data to corrupt
            rows_to_corrupt (list): the rows to corrupt
            numeric_column_to_corrupt (str): the column name to corrupt

        Returns:
            pd.DataFrame: the `data_to_corrupt`, after applying the corurption on top of it
        """
        from synqtab.errors import Placeholder
        replacement = Placeholder.NUMERIC_MISSING_VALUE
        data_to_corrupt.loc[rows_to_corrupt, numeric_column_to_corrupt] = replacement
        return data_to_corrupt

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt, **kwargs):
        import pandas as pd
        from synqtab.reproducibility import ReproducibleOperations
        
        # TODO ensure that the corrupted rows are correct, now they are not
        
        # hold out a copy of the original rows that are going to be duplicated
        original_duplicate_rows = data_to_corrupt.loc[rows_to_corrupt].copy(deep=True)
        
        # apply corruption to the dataframe to produce the near duplicates in-place
        for column_to_corrupt in columns_to_corrupt:
            if column_to_corrupt in self.categorical_columns:
                # for categorical columns, insert typos
                data_to_corrupt = self._apply_corruption_to_categorical_column(
                    data_to_corrupt, rows_to_corrupt, column_to_corrupt, **kwargs
                )
                continue

            # for numeric columns, insert implicit missing values
            data_to_corrupt = self._apply_corruption_to_numeric_column(
                data_to_corrupt, rows_to_corrupt, column_to_corrupt, **kwargs
            )

        # add back the hold-out rows so that the corrupted rows co-exist with the original ones, creating near duplicates
        data_to_corrupt = pd.concat([data_to_corrupt, original_duplicate_rows])
        data_to_corrupt = ReproducibleOperations.shuffle_reindex_dataframe(data_to_corrupt)
        return data_to_corrupt
