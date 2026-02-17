from typing import Self

from synqtab.experiments.Experiment import Experiment
from synqtab.utils import get_logger


LOG = get_logger(__file__)


class NormalExperiment(Experiment):
    
    @classmethod
    def short_name(cls):
        from synqtab.enums import ExperimentType
        return str(ExperimentType.NORMAL)
    
    def _run(self) -> None:
        from synqtab.data import PostgresClient, MinioClient
        from synqtab.enums import MinioBucket, ProblemType, DataPerfectness
        from synqtab.environment import MAX_TRAINING_ROWS
        from synqtab.mappings.mappings import GENERATOR_MODEL_TO_GENERATOR_INSTANCE
        from synqtab.reproducibility import ReproducibleOperations
        from synqtab.utils import timed_computation

        LOG.info(f"Entering the _run() function of Normal Experiment {str(self)}")
        
        real_perfect_df = self.dataset._fetch_real_perfect_dataframe()
        target_column_name = self.dataset.target_feature
        target = real_perfect_df[target_column_name]
        problem_type = ProblemType(self.dataset.problem_type)
        training_df, validation_df = ReproducibleOperations.train_test_split(
            real_perfect_df, test_size=0.5, stratify=target, problem_type=problem_type)
        
        if len(training_df) > MAX_TRAINING_ROWS:
            from synqtab.data import PostgresClient
            LOG.info(f"Experiment {str(self)} will be skipped, because the dataset has {len(training_df)} rows.")
            PostgresClient.write_skipped_computation(computation_id=str(self), reason=f"More than {MAX_TRAINING_ROWS} rows ({len(training_df)}).")
            self._should_compute = False
            return
        
        corrupted_rows = corrupted_cols = []
        if self.data_error:
            if self.data_error_rate:
                data_error_instance = self.data_error.get_class()(row_fraction=self.data_error_rate)
                training_df, corrupted_rows, corrupted_cols = data_error_instance.corrupt(
                    data=training_df,
                    categorical_columns=self.dataset.categorcal_features,
                    target_column=self.dataset.target_feature,
                )
                LOG.info(f"Data Corruption was completed successfully for experiment {str(self)}")
                
                if len(corrupted_cols) == 0:
                    from synqtab.data import PostgresClient
                    LOG.info(f"Experiment {str(self)} will be skipped, because no columns to corrupt were found.")
                    LOG.info(f"Experiment {str(self)}. Categorical: {self.dataset.categorcal_features}, All: {training_df.columns}, Error Applicability: {data_error_instance.data_error_applicability()}.")
                    self._should_compute = False
                    PostgresClient.write_skipped_computation(computation_id=str(self), reason="No columns to corrupt.")
                    return

                if self.data_perfectness == DataPerfectness.SEMIPERFECT:
                  training_df.drop(corrupted_rows)
        
        LOG.info(f"Initializing {self.generator} generator for experiment {str(self)}")
        y = training_df[target_column_name]
        X = training_df.drop(columns=[target_column_name])
        generator_instance = GENERATOR_MODEL_TO_GENERATOR_INSTANCE.get(self.generator)
        
        synthetic_df, elapsed_time = timed_computation(
            computation=generator_instance.generate,
            params={
                'X_initial': X,
                'y_initial': y,
                'n_samples': len(X),
                'metadata': self.dataset.metadata,
            }
        )
        LOG.info(f"Generation for experiment {str(self)} was completed in {elapsed_time} seconds.")

        # Action 1: Write the Synthetic data to MinIO for asynchronous evaluation
        MinioClient.upload_dataframe_as_parquet_to_bucket(
            df=synthetic_df,
            bucket_name=MinioBucket.SYNTHETIC,
            object_name=self.minio_path()
        )
        LOG.info(f"Successfully wrote the synthetic data of experiment {str(self)} to MinIO '{self.minio_path()}'.")
        
        # Action 2: Write experiment metadata to Postgres for offline analysis
        import json
    
        corrupted_rows = corrupted_rows.tolist() if 'numpy' in str(type(corrupted_rows)) else corrupted_rows
        corrupted_cols = corrupted_cols.tolist() if 'numpy' in str(type(corrupted_cols)) else corrupted_cols

        PostgresClient.write_experiment(
            experiment_id=str(self),
            experiment_type=self.short_name(),
            dataset_name=self.dataset.dataset_name,
            random_seed=str(ReproducibleOperations.get_current_random_seed()),
            data_perfectness=str(self.data_perfectness),
            data_error=str(self.data_error) if self.data_error else None,
            error_rate=str(int(self.data_error_rate * 100)) if self.data_error_rate else None,
            generator=str(self.generator),
            training_size=str(len(X)),
            synthetic_size=str(len(synthetic_df)),
            corrupted_rows=json.dumps(corrupted_rows),
            corrupted_cols=json.dumps(corrupted_cols),
            execution_time=elapsed_time,
        )
        LOG.info(f"Successfully wrote the metadata of experiment {str(self)} to Postgres.")


    def _publish_tasks(self) -> Self:
        # TODO FIND A WAY TO POPULATE THE PARAMS AS THE SDMETRICS ARE EXPECTING TO GET THESE
        params = dict()
        
        from synqtab.enums import SINGULAR_EVALUATORS, DUAL_EVALUATORS
        from synqtab.evaluators import Evaluation
        from synqtab.mappings import (
            EVALUATION_METHOD_TO_EVALUATION_CLASS,
            SINGULAR_EVALUATION_TARGETS, DUAL_EVALUATION_TARGETS
        )
        
        published_tasks = 0
        skipped_tasks = 0
        for evaluation_method in self.evaluators:
            
            evaluation_pairs = []
            if evaluation_method in SINGULAR_EVALUATORS:
                evaluation_pairs = SINGULAR_EVALUATION_TARGETS
            elif evaluation_method in DUAL_EVALUATORS:
                evaluation_pairs = DUAL_EVALUATION_TARGETS
            else:
                raise ValueError(f"Evaluation method {str(evaluation_method)} was not found in neither the singular nor dual evaluators.")
            
            for evaluation_pair in evaluation_pairs:
                evaluation = Evaluation(
                    *evaluation_pair,
                    experiment=self,
                    evaluation_method=evaluation_method,
                )
                was_published = evaluation.publish_task_if_valid()
                if was_published:
                    published_tasks += 1
                else:
                    # can happen if the evaluation is not valid, e.g., R2 evaluation on classification dataset
                    skipped_tasks += 1
                
        LOG.info(f"Successfully published {published_tasks} and skipped {skipped_tasks} tasks for experiment {str(self)}")        
        return self