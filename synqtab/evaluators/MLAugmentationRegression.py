from interface_meta import override
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from synqtab.datasets import Dataset
from synqtab.evaluators.DualEvaluator import DualEvaluator
from sdmetrics.single_table.data_augmentation import BinaryClassifierRecallEfficacy

class MLAugmentationRegression(DualEvaluator):
    def __init__(self, target_label:str, problem_type:str, notes:bool = False):
        self.notes = notes
        self.target_label = target_label
        self.problem_type = problem_type

    @override
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict[str, Any], holdout_table:Optional[pd.DataFrame]) -> Any:
        # Custom implementation for regression data augmentation
        # Train on real data only (baseline)
        X_real = real_data.drop(columns=[self.target_label])
        y_real = real_data[self.target_label]

        # Train on augmented data (real + synthetic)
        augmented_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        X_augmented = augmented_data.drop(columns=[self.target_label])
        y_augmented = augmented_data[self.target_label]

        # Validation data
        X_val = holdout_table.drop(columns=[self.target_label])
        y_val = holdout_table[self.target_label]

        # Train baseline model (real data only)
        baseline_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        baseline_model.fit(X_real, y_real)
        baseline_pred = baseline_model.predict(X_val)
        baseline_r2 = r2_score(y_val, baseline_pred)
        baseline_mse = mean_squared_error(y_val, baseline_pred)
        baseline_mae = mean_absolute_error(y_val, baseline_pred)

        # Train augmented model (real + synthetic data)
        augmented_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        augmented_model.fit(X_augmented, y_augmented)
        augmented_pred = augmented_model.predict(X_val)
        augmented_r2 = r2_score(y_val, augmented_pred)
        augmented_mse = mean_squared_error(y_val, augmented_pred)
        augmented_mae = mean_absolute_error(y_val, augmented_pred)

        # Compute improvement score (normalized between 0 and 1)
        # Higher R2 is better, so positive improvement is good
        r2_improvement = augmented_r2 - baseline_r2
        # Lower MSE is better, so we want negative change
        mse_improvement = baseline_mse - augmented_mse
        # Lower MAE is better
        mae_improvement = baseline_mae - augmented_mae

        # Normalize the score: if augmented performs better, score > 0.5, otherwise < 0.5
        # Use R2 as primary metric since it's already normalized
        score = 0.5 + (r2_improvement / 2)  # Maps improvement to [0, 1] range
        score = np.clip(score, 0, 1)  # Ensure within bounds

        if not self.notes:
            return {"score": score}
        else:
            return {
                "score": score,
                "notes": {
                    "augmented_data": {
                        "r2": augmented_r2,
                        "mse": augmented_mse,
                        "mae": augmented_mae
                    },
                    "real_data_baseline": {
                        "r2": baseline_r2,
                        "mse": baseline_mse,
                        "mae": baseline_mae
                    },
                    "parameters": {
                        "r2_improvement": r2_improvement,
                        "mse_improvement": mse_improvement,
                        "mae_improvement": mae_improvement
                    }
                }
            }



if __name__ == "__main__":
    # Example usage
    prior_config = Dataset(dataset_name="miami_housing",
                           mode="minio")

    prior = prior_config.fetch_prior_dataset()
    sd_metadata = prior_config.create_sdmetrics_metadata()
    prior_info = prior_config.get_config()

    evaluator = MLAugmentationRegression(prior_info['target_feature'], prior_info["problem_type"], notes=True)
    results = evaluator.evaluate(prior, prior, metadata=sd_metadata, holdout_table=prior)
    print(results)