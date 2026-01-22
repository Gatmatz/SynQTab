from enum import Enum

class Metadata(Enum):
    NAME = 'name'
    PROBLEM_TYPE = 'problem_type'
    TARGET_FEATURE = 'target_feature'
    CATEGORICAL_FEATURES = 'categorical_features'


class ProblemType(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class DataPerfectness(Enum):
    PERFECT = 'perfect'
    IMPERFECT = 'imperfect'
    SEMIPERFECT = 'semiperfect'
    
    def shortname(self) -> str:
        """First 3 letters in capital for imperfect; first 4 letters in capital for
        anything else.

        Returns:
            str: The shortname of the DataPerfectness object.
        """
        if self == DataPerfectness.IMPERFECT:
            return self.value.upper()[:3]
        return self.value.upper()[:4]
    

class DataErrorType(Enum):
    CATEGORICAL_SHIFT = 'categorical_shift'
    GAUSSIAN_NOISE = 'gaussian_noise'
    PLACEHOLDER = 'placeholder'
    NEAR_DUPLICATE = 'near_duplicate'
    OUTLIER = 'outlier'
    INCONSISTENCY = 'inconsistency'
    LABEL_ERROR = 'label_error'

