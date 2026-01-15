from enum import Enum


class DataErrorApplicability(Enum):
    ANY_COLUMN = "Numeric and Categorical Columns"
    NUMERIC_ONLY = "Numeric Columns Only"
    CATEGORICAL_ONLY = "Categorical Columns Only"
