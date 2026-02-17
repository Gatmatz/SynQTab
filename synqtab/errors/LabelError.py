from synqtab.errors.CategoricalShift import CategoricalShift
from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations


class LabelError(CategoricalShift):

    def data_error_applicability(self):
        from synqtab.errors import DataErrorApplicability
        
        return DataErrorApplicability.CATEGORICAL_ONLY
    
    def short_name(self):
        from synqtab.enums import DataErrorType
        
        return DataErrorType.LABEL_ERROR
    
    def full_name(self):
        return "Label error"

    def _apply_corruption(self, data_to_corrupt, rows_to_corrupt, columns_to_corrupt, **kwargs):
        from synqtab.enums.data import Metadata
        
        # we override the columns to corrupt to only corrupt the target column
        self.columns_to_corrupt = [kwargs.get('target_column')]
        
        return super()._apply_corruption(
            data_to_corrupt=data_to_corrupt,
            rows_to_corrupt=rows_to_corrupt,
            columns_to_corrupt=self.columns_to_corrupt,
            **kwargs
        )
