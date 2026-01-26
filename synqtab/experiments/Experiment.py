from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Self

from synqtab.data.Dataset import Dataset
from synqtab.enums import (
    DataPerfectness, DataErrorType,
    EvaluationMethod, GeneratorModel
)
from synqtab.utils import get_logger


LOG = get_logger(__file__)

import logging
# logging.disable(20)


class Experiment(ABC):
    
    _delimiter: str = '#'
    _NULL: str = 'NULL'
    translator: str

    def __init__(
        self,
        dataset: Dataset,
        generator: GeneratorModel,
        data_error_type: Optional[DataErrorType] = None,
        data_error_rate: Optional[float] = None,
        data_perfectness: DataPerfectness = DataPerfectness.PERFECT,
        evaluation_methods: Optional[list[EvaluationMethod]] = None,
        options: Optional[Dict[Any, Any]] = None,
    ):
        self.dataset = dataset
        self.generator = generator
        self.data_error = data_error_type
        self.data_error_rate = data_error_rate
        self.data_perfectness = data_perfectness
        self.evaluators = evaluation_methods
        self.options = options
        
        self._should_compute = (not self._exists_in_postgres())
        
        self.training_X = None
        y = None
        self.synthetic_df = None
        
    @classmethod
    @abstractmethod
    def short_name(cls):
        pass
    
    @abstractmethod
    def _run(self) -> None:
        """
        Perform the actual experimental run and any writing options to Postgres and/or MinIO.
        All validations have been verified before reaching this function,
        so you can freely assume that the experiment must run.
        """
        pass
        
    def even_if_exists(self) -> Self:
        LOG.info(f"Experiment {str(self)} is set to be force-computed, even if it exists in Postgres.")
        self._should_compute = True
        return self
    
    # IMPORTANT: Keep this method aligned with the from_str() method!
    def _get_experiment_id_parts(self):
        from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations
        return [
            str(self.short_name()),  # Experiment shortname, e.g., 'NOR' for Normal Experiment
            str(self.dataset.dataset_name),  # Dataset name, e.g., 'anneal'
            str(ReproducibleOperations.get_current_random_seed()),  # Random seed
            str(self.data_perfectness), # Data perfectness level, e.g., 'PERF' for perfect
            str(self.data_error) if self.data_error else self._NULL,    # Data error type, e.g., 'OUT' for outliers
            str(int(self.data_error_rate * 100)) if self.data_error_rate else self._NULL, # Data error rate multiplied by 100, e.g., 0.2 -> 20 -> '20'
            str(self.generator),   # Generator type, e.g., 'tabpfn' 
        ]
    
    # IMPORTANT: Keep this method aligned with the _get_experiment_parts() method!
    @classmethod
    def from_str(cls, experiment_id: str) -> tuple[Self, int]:
        from synqtab.mappings.mappings import EXPERIMENT_TYPE_TO_EXPERIMENT_CLASS
        from synqtab.data.Dataset import Dataset
        from synqtab.enums.data import DataPerfectness
        from synqtab.enums.generators import GeneratorModel
        
        experiment_id_parts = experiment_id.split(cls._delimiter)
        experiment_short_name = experiment_id_parts[0]
        dataset = Dataset(experiment_id_parts[1])
        random_seed = int(experiment_id_parts[2])
        data_perfectness = DataPerfectness(experiment_id_parts[3])
        data_error = None if experiment_id_parts[4] == cls._NULL else DataErrorType(experiment_id_parts[4])
        data_error_rate = None if experiment_id_parts[5] == cls._NULL else float(int(experiment_id_parts[5]) / 100)
        generator = GeneratorModel(experiment_id_parts[6])
        
        for experiment_type, experiment_class in EXPERIMENT_TYPE_TO_EXPERIMENT_CLASS.items():
            if experiment_short_name == experiment_class.short_name():
                LOG.info(f"Experiment {experiment_id} found to be of class {experiment_class.__name__}")
                return experiment_class(
                    dataset=dataset,
                    generator=generator,
                    data_error_type=data_error,
                    data_error_rate=data_error_rate,
                    data_perfectness=data_perfectness,
                ), random_seed # RETURNS TUPLE: Experiment class + random seed!
    
    def __str__(self):
        experiment_id_parts = self._get_experiment_id_parts()
        return self._delimiter.join(experiment_id_parts)
    
    def minio_path(self):
        from synqtab.enums import MinioFolder
        
        experiment_id_parts = self._get_experiment_id_parts()
        return MinioFolder.create_prefix(
            MinioFolder.DATA, *experiment_id_parts, ignore=self._NULL
        )
        
    def run(self) -> Self:
        if not self._should_compute:
            from synqtab.data import PostgresClient
            LOG.info(f"Running experiment {str(self)} will be skipped because it already exists in Postgres.")
            PostgresClient.write_skipped_computation(computation_id=str(self), reason="Already exists in Postgres.")
            return self
        
        self._run()
        return self
    
    def populate_tasks(self) -> Self:
        if not self._should_compute:
            LOG.info(f"Populating tasks for experiment {str(self)} will be skipped because the experiment already exists in Postgres.")
            return self
        
        # TODO FIND A WAY TO POPULATE THE PARAMS AS THE SDMETRICS ARE EXPECTING TO GET THESE
        params = dict()
        
        from synqtab.data import MinioClient
        from synqtab.enums import SINGULAR_EVALUATORS, DUAL_EVALUATORS
        from synqtab.mappings import EVALUATION_METHOD_TO_EVALUATION_CLASS
        for evaluation_method in self.evaluators:
            evaluator_instance = EVALUATION_METHOD_TO_EVALUATION_CLASS.get(evaluation_method)(params=params)
            if evaluator_instance in SINGULAR_EVALUATORS:
                # TODO POPULATE 4 TASKS, ONE FOR EACH OF THE TABLES see self._populate_task
                pass # OR PROBABLY RETURN
            
            if evaluator_instance in DUAL_EVALUATORS:
                # TODO POPULATE 5 TASKS AS AGREED see self._populate_task
                pass
            
        # TODO DO THE JOB HERE
        return self
    
    def _populate_task(self) -> None:
        pass

    def _exists_in_postgres(self) -> bool:
        from synqtab.data import PostgresClient
        
        return PostgresClient.experiment_exists(str(self))
    