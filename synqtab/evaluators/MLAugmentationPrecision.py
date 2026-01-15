from interface_meta import override
from typing import Any, Dict, Optional
import pandas as pd

from synqtab.datasets import Dataset
from synqtab.evaluators.DualEvaluator import DualEvaluator
from sdmetrics.single_table.data_augmentation import BinaryClassifierPrecisionEfficacy

class MLAugmentationPrecision(DualEvaluator):
    def __init__(self, target_label:str, positive_class, notes:bool = False):
        self.notes = notes
        self.target_label = target_label
        self.positive_class = positive_class

    @override
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict[str, Any], holdout_table:Optional[pd.DataFrame]) -> Any:
        score = BinaryClassifierPrecisionEfficacy.compute_breakdown(
            real_training_data=real_data,
            synthetic_data=synthetic_data,
            real_validation_data=holdout_table,
            minority_class_label=self.positive_class,
            metadata=metadata,
            prediction_column_name=self.target_label,

        )
        if not self.notes:
            return {"precision": score['score']}
        else:
            return {
                "precision": score['score'],
                "notes": {
                    "augmented_data": score['augmented_data'],
                    "real_data_baseline": score['real_data_baseline'],
                    "parameters": score['parameters']
                }
            }

if __name__ == "__main__":
    # Example usage
    prior_config = Dataset(dataset_name="blood-transfusion-service-center",
                           mode="minio")

    prior = prior_config.fetch_prior_dataset()
    sd_metadata = prior_config.create_sdmetrics_metadata()
    prior_info = prior_config.get_config()

    evaluator = MLAugmentationPrecision(prior_info['target_feature'], 1, notes=True)
    results = evaluator.evaluate(prior, prior, metadata=sd_metadata, holdout_table=prior)
    print(results)