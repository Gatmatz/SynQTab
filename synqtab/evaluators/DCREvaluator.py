from typing import override, Any, Dict

import pandas as pd
from sdmetrics.single_table import DCROverfittingProtection
from synqtab.datasets import Dataset
from synqtab.evaluators.DualEvaluator import DualEvaluator


class DCREvaluator(DualEvaluator):

    def __init__(self, notes: bool = False):
        self.notes = notes

    @override
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict[str, Any], holdout_table:pd.DataFrame) -> Any:
        score = DCROverfittingProtection.compute_breakdown(
            real_training_data=real_data,
            synthetic_data=synthetic_data,
            real_validation_data=holdout_table,
            metadata=metadata
        )
        if self.notes:
            return {
                'dcr_score': score.get('score'),
                'notes': {
                    'synthetic_data_percentages': score.get('synthetic_data_percentages'),
                    'real_validation_data_percentages': score.get('real_validation_data_percentages'),
                }
            }
        else:
            return {'dcr_score': score.get('score')}

if __name__ == "__main__":
    # Example usage
    prior_config = Dataset(dataset_name="blood-transfusion-service-center",
                           mode="minio")

    prior = prior_config.fetch_prior_dataset()
    sd_metadata = prior_config.create_sdmetrics_metadata()

    evaluator = DCREvaluator(notes=True)
    results = evaluator.evaluate(prior, prior, sd_metadata, prior)
    print(results)