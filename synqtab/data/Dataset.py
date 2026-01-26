from typing import Any, Optional

import pandas as pd

from synqtab.enums import DataPerfectness, ExperimentType
from synqtab.errors import DataError
from synqtab.utils import get_logger


LOG = get_logger(__file__)


class Dataset:
    
    _PREPARE_FUNCTION = 'prepare'
    _POLLUTE_FUNCTION = 'pollute'
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.metadata = self._fetch_metadata()
        self.problem_type = self.metadata.get('problem_type', 'classification')
        self.target_feature = self.metadata.get('target_feature')
        self.categorcal_features = self.metadata.get('categorical_features', [])
        
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
    
    def _no_post_processing(self, *input):
        return input
        
    def _get_experiment_type_to_functions_mapping(self):
        return {
            ExperimentType.NORMAL: {
                self._PREPARE_FUNCTION: self._split_data_for_normal_experiment,
                self._POLLUTE_FUNCTION: self._pollute_data_for_normal_experiment,
            },
            ExperimentType.PRIVACY: {
                self._PREPARE_FUNCTION: self._split_data_for_privacy_experiment,
                self._POLLUTE_FUNCTION: self._pollute_data_for_privacy_experiment,
            },
            ExperimentType.AUGMENTATION: {
                self._PREPARE_FUNCTION: self._split_data_for_augmentation_experiment,
                self._POLLUTE_FUNCTION: self._pollute_data_for_augmentation_experiment,
            },
            ExperimentType.REBALANCING: {
                self._PREPARE_FUNCTION: self._split_data_for_rebalancing_experiment,
                self._POLLUTE_FUNCTION: self._pollute_data_for_rebalancing_experiment,
            }
        }
    
    def _get_perfectness_to_functions_mapping(self):
        return {
            DataPerfectness.PERFECT: self._no_post_processing,
            DataPerfectness.IMPERFECT: self._no_post_processing,
            DataPerfectness.SEMIPERFECT: self._construct_semi_perfect_from_imperfect,
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
        return MinioClient.read_parquet_from_bucket(
            bucket_name=bucket_name,
            object_name=object_name
        )
    
    def _prepare_data_for_experiment(
        self,
        df: pd.DataFrame,
        experiment_type: ExperimentType,
        data_perfectness: Optional[DataPerfectness] = DataPerfectness.PERFECT,
        data_error: Optional[DataError] = None
    ) -> dict[str, Any]:
        
        experiment_type_to_functions_mapping = self._get_experiment_type_to_functions_mapping()
        
        if not experiment_type in experiment_type_to_functions_mapping:
            error_message = f"Unknown experiment type {experiment_type} for dataset {self.dataset_name}."
            LOG.error(error_message)
            raise NotImplementedError(error_message)
        
        functions = experiment_type_to_functions_mapping.get(experiment_type)
        splitting_function = functions.get(self._PREPARE_FUNCTION)
        pollution_function = functions.get(self._POLLUTE_FUNCTION)
        
        splitted_dataframes = splitting_function(df)
        if not data_error or data_perfectness == DataPerfectness.PERFECT:
            return splitted_dataframes
        
        polluted_dataframes = pollution_function(splitted_dataframes)
         
        data_perfectness_to_postprocessing_function_mapping = self._get_experiment_type_to_functions_mapping()
        if not data_perfectness in data_perfectness_to_postprocessing_function_mapping:
            error_message = f"Unknown perfectness type {data_perfectness} for dataset {self.dataset_name}."
            LOG.error(error_message)
            raise NotImplementedError(error_message)
        
        postprocessing_function = data_perfectness_to_postprocessing_function_mapping.get(data_perfectness)
        return postprocessing_function(polluted_dataframes)    
        
    def _split_data_for_normal_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass
    
    def _split_data_for_privacy_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass
    
    def _split_data_for_augmentation_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass

    def _split_data_for_rebalancing_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass
    
    ########
    def _pollute_data_for_normal_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass
    
    def _pollute_data_for_privacy_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass
    
    def _pollute_data_for_augmentation_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass

    def _pollute_data_for_rebalancing_experiment(self, df: pd.DataFrame) -> dict[str, Any]:
        pass
    
    def _construct_semi_perfect_from_imperfect(self, params: dict[str, Any]) -> dict[str, Any]:
        pass
        
    def fetch_split_data_for_experiment(
        self,
        experiment_type: ExperimentType,
        data_perfectness: DataPerfectness = DataPerfectness.PERFECT,
        data_error: Optional[DataError] = None,
    ) -> dict[str, Any]:
        
        if not data_perfectness:
            data_perfectness = DataPerfectness.PERFECT
        
        if not data_error and data_perfectness != DataPerfectness.PERFECT:
            LOG.error(
                f"Unable to prepare {data_perfectness} data for {self.dataset_name} without a data error."
            )
            raise ValueError(f"{data_perfectness} data requires a data error, but none was specified.")
        
        if data_error and data_perfectness == DataPerfectness.PERFECT:
            LOG.error(
                f"Ambiguous request for dataset {self.dataset_name}. Cannot prepare \
                    perfect data and at the same time pollute with {data_error.full_name()}."
            )
            raise ValueError(f"Perfect data is not compatible with data errors at the same time.")
        
        
        real_perfect_df = self._fetch_real_perfect_dataframe()
        return self._prepare_data_for_experiment(real_perfect_df, experiment_type, data_perfectness, data_error)
    
    