from typing import Dict, Any

from synqtab.datasets import Dataset
from synqtab.evaluators.DualEvaluator import DualEvaluator
import pandas as pd


class MLEfficacy(DualEvaluator):
    def __init__(self, problem_type:str, target_column:str, notes: bool = False):
        self.problem_type = problem_type
        self.target_column = target_column
        self.notes = notes

    def _find_classification_type(self, data: pd.DataFrame, target_column: str) -> str:
        unique_values = data[target_column].nunique()
        if unique_values == 2:
            return "binary"
        else:
            return "multiclass"

    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict[str, Any], holdout_table:pd.DataFrame = None) -> Any:
        scores = {}
        if self.problem_type=="regression":
            from sdmetrics.single_table import LinearRegression

            scores = LinearRegression.compute(
                test_data=real_data,
                train_data=synthetic_data,
                target=self.target_column,
                metadata=metadata
            )
        elif self.problem_type=="classification":
            type = self._find_classification_type(synthetic_data, self.target_column)
            if type=="binary":
                from sdmetrics.single_table import BinaryAdaBoostClassifier

                scores = BinaryAdaBoostClassifier.compute(
                    test_data=real_data,
                    train_data=synthetic_data,
                    target=self.target_column,
                    metadata=metadata
                )
            else:
                from sdmetrics.single_table import MulticlassDecisionTreeClassifier

                scores = MulticlassDecisionTreeClassifier.compute(
                    test_data=real_data,
                    train_data=synthetic_data,
                    target=self.target_column,
                    metadata=metadata
                )
        return {'accuracy': scores}

if __name__ == "__main__":
    # Example usage
    prior_config = Dataset(dataset_name="blood-transfusion-service-center",
                    mode="minio")

    prior = prior_config.fetch_prior_dataset()
    sd_metadata = prior_config.create_sdmetrics_metadata()
    prior_info = prior_config.get_config()

    evaluator = MLEfficacy(prior_info['problem_type'], prior_info['target_feature'], notes=True)
    results = evaluator.evaluate(prior, prior, metadata=sd_metadata)
    print(results)