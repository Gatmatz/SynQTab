from synqtab.errors import DataError
from synqtab.enums.EasilyStringifyableEnum import EasilyStringifyableEnum


class Metadata(EasilyStringifyableEnum):
    NAME = 'name'
    PROBLEM_TYPE = 'problem_type'
    TARGET_FEATURE = 'target_feature'
    CATEGORICAL_FEATURES = 'categorical_features'


class ProblemType(EasilyStringifyableEnum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class DataPerfectness(EasilyStringifyableEnum):
    PERFECT = 'PERF'
    IMPERFECT = 'IMP'
    SEMIPERFECT = 'SEMI'
    
    def short_name(self) -> str:
        """First 3 letters in capital for imperfect; first 4 letters in capital for
        anything else.

        Returns:
            str: The shortname of the DataPerfectness object.
        """
        if self == DataPerfectness.IMPERFECT:
            return self.value.upper()[:3]
        return self.value.upper()[:4]
    

class DataErrorType(EasilyStringifyableEnum):
    CATEGORICAL_SHIFT = 'SFT'
    GAUSSIAN_NOISE = 'NOI'
    PLACEHOLDER = 'PLC'
    NEAR_DUPLICATE = 'DUP'
    OUTLIER = 'OUT'
    INCONSISTENCY = 'INC'
    LABEL_ERROR = 'LER'
    
    def get_class(self) -> DataError.__class__:
        
        match(self):
            case DataErrorType.CATEGORICAL_SHIFT:
                from synqtab.errors import CategoricalShift
                return CategoricalShift
            case DataErrorType.GAUSSIAN_NOISE:
                from synqtab.errors import GaussianNoise
                return GaussianNoise
            case DataErrorType.INCONSISTENCY:
                from synqtab.errors import Inconsistency
                return Inconsistency
            case DataErrorType.NEAR_DUPLICATE:
                from synqtab.errors import NearDuplicateRow
                return NearDuplicateRow
            case DataErrorType.OUTLIER:
                from synqtab.errors import Outliers
                return Outliers
            case DataErrorType.PLACEHOLDER:
                from synqtab.errors import Placeholder
                return Placeholder
