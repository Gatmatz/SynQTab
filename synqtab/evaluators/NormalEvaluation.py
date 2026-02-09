from synqtab.evaluators import Evaluation
from synqtab.utils import get_logger


LOG = get_logger(__file__)


class NormalEvaluation(Evaluation):
    
    def _run(self):
        from synqtab.data import PostgresClient
        from synqtab.enums import ProblemType, DataPerfectness, EvaluationInput
        from synqtab.mappings.mappings import EVALUATION_METHOD_TO_EVALUATION_CLASS
        from synqtab.reproducibility import ReproducibleOperations
        from synqtab.utils import timed_computation
        
        evaluation_full_name = str(self) + '/' + str(self.experiment)
        

        LOG.info(f"Entering the _run() function of Normal Evaluation {evaluation_full_name}")
        
        real_perfect_df = self.experiment.dataset._fetch_real_perfect_dataframe()
        target_column_name = self.experiment.dataset.target_feature
        target = real_perfect_df[target_column_name]
        problem_type = ProblemType(self.experiment.dataset.problem_type)
        sdmetrics_metadata = self.experiment.dataset.get_sdmetrics_single_table_metadata()
        training_df, validation_df = ReproducibleOperations.train_test_split(
            real_perfect_df, test_size=0.5, stratify=target, problem_type=problem_type)
        
        corrupted_rows = corrupted_cols = []
        if self.data_error:
            if self.data_error_rate:
                data_error_instance = self.experiment.data_error.get_class()(row_fraction=self.experiment.data_error_rate)
                training_df, corrupted_rows, corrupted_cols = data_error_instance.corrupt(
                    data=training_df,
                    categorical_columns=self.experiment.dataset.categorcal_features,
                    target_column=self.experiment.dataset.target_feature,
                )
                LOG.info(f"Data Corruption was completed successfully for experiment {str(self)}")

                if self.data_perfectness == DataPerfectness.SEMIPERFECT:
                  training_df.drop(corrupted_rows)
                  
        params = {
            EvaluationInput.PROBLEM_TYPE: str(problem_type),
            EvaluationInput.METADATA: sdmetrics_metadata,
            EvaluationInput.REAL_VALIDATION_DATA: validation_df,
            EvaluationInput.NOTES: True,
            EvaluationInput.PREDICTION_COLUMN_NAME: target_column_name,
            EvaluationInput.KNOWN_COLUMN_NAMES: list(training_df.columns),
            EvaluationInput.SENSITIVE_COLUMN_NAMES: [],
            EvaluationInput.DATA: "TODO",
            EvaluationInput.MINORITY_CLASS_LABEL: "TODO",
            EvaluationInput.SYNTHETIC_DATA: "TODO",
            EvaluationInput.REAL_TRAINING_DATA: "TODO",
        }
        
        evaluator_instance = EVALUATION_METHOD_TO_EVALUATION_CLASS.get(self.evaluation_method)(params)
        (result, notes), elapsed_time = timed_computation(
            computation=evaluator_instance.evaluate,
            params=dict(),
        )
        
        import json
        PostgresClient.write_evaluation_result(
            evaluation_id=str(self),
            first_target=str(self.evaluation_targets[0]),
            second_target=str(self.evaluation_targets[1]) if len(self.evaluation_targets) > 1 else None,
            result=result,
            execution_time=elapsed_time,
            notes=json.dumps(notes)
        )
