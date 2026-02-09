from typing import Any, Optional

import pandas as pd

from synqtab.enums import DataPerfectness, ExperimentType, Metadata
from synqtab.errors import DataError
from synqtab.utils import get_logger


LOG = get_logger(__file__)


class Dataset:
    
    _PREPARE_FUNCTION = 'prepare'
    _POLLUTE_FUNCTION = 'pollute'
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.metadata = self._fetch_metadata()
        self.problem_type = self.metadata.get(str(Metadata.PROBLEM_TYPE), 'classification')
        self.target_feature = self.metadata.get(str(Metadata.TARGET_FEATURE))
        self.categorcal_features = self.metadata.get(str(Metadata.CATEGORICAL_FEATURES), [])
        
    def get_metadata(self) -> dict:
        """
        Returns the metadata of the dataset. See synqtab.enums.data.py -> Metadata enum
        for more information on the available options.

        Returns:
            dict: A dictionary containing metadata about the dataset.
        """
        from synqtab.enums.data import Metadata
        return {
            Metadata.NAME: self.dataset_name,
            Metadata.PROBLEM_TYPE: self.problem_type,
            Metadata.TARGET_FEATURE: self.target_feature,
            Metadata.CATEGORICAL_FEATURES: self.categorical_features
    }

    def _fetch_metadata(self):
        from synqtab.data import MinioClient
        from synqtab.enums import MinioBucket, MinioFolder
        
        bucket_name = MinioBucket.REAL.value
        object_name = MinioFolder.create_prefix(
            MinioFolder.PERFECT,
            MinioFolder.METADATA,
            f"{self.dataset_name}.yaml"
        )
        return MinioClient.read_yaml_from_bucket(
            bucket_name=bucket_name,
            object_name=object_name
        )
        
    def _fetch_real_perfect_dataframe(self) -> pd.DataFrame:
        from synqtab.data import MinioClient
        from synqtab.enums import MinioBucket, MinioFolder
        
        bucket_name = MinioBucket.REAL.value
        object_name = MinioFolder.create_prefix(
            MinioFolder.PERFECT,
            MinioFolder.DATA,
            f"{self.dataset_name}.parquet"
        )

        df = MinioClient.read_parquet_from_bucket(
            bucket_name=bucket_name,
            object_name=object_name
        )

        # ensure that the categorical columns are explicitly declared in the pd DataFrame
        for column in self.categorcal_features:
            if column in df.columns:
                df[column] = df[column].astype('category')

        return df

    def get_sdmetrics_single_table_metadata(self) -> dict[str, Any]:
        """Example based on https://docs.sdv.dev/sdmetrics/getting-started/metadata/single-table-metadata
        {
            "columns": {
                "age": {
                    "sdtype": "numerical"
                }
                "tier": {
                    "sdtype": "categorical"
                },
                "paid_amt": {
                    "sdtype": "numerical"
                },
            }
        }

        Returns:
            dict[str, Any]: An sdmetrics metadata dictionary
        """
        from synqtab.enums import ProblemType
        
        columns_dict: dict[str, str] = dict()
        all_columns = self._fetch_real_perfect_dataframe().columns
        
        # Create one sub-dictionary per feature column
        for column in all_columns:
            if column in self.categorcal_features:
                columns_dict[column] = {"sdtype": "categorical"}
            else:
                columns_dict[column] = {"sdtype": "numerical"}

        # create one sub-dictionary for the target column
        if self.problem_type == str(ProblemType.CLASSIFICATION):
            columns_dict[self.target_feature] = {"sdtype": "categorical"}
        else:
            columns_dict[self.target_feature] = {"sdtype": "numerical"}

        # nest the columns dict inside a "columns" key according to the sdmetrics expected format
        return {"columns": columns_dict}
