import pandas as pd
from sklearn.ensemble import IsolationForest

from synqtab.datasets import Dataset
from synqtab.evaluators.SingleEvaluator import SingleEvaluator


class IsolationForestEvaluator(SingleEvaluator):
    def __init__(self, n_estimators=100, contamination='auto', random_state=42, notes:bool = False):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.notes = notes

    def evaluate(self, data_1: pd.DataFrame) -> dict:
        iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )

        # Fit and predict (-1 for outliers, 1 for inliers)
        predictions = iso_forest.fit_predict(data_1)

        # Get anomaly scores (more negative = more outlier-like)
        scores = iso_forest.score_samples(data_1)

        # Create result dictionary
        if self.notes is False:
            return {'n_outliers': int((predictions == -1).sum())}
        else:
            result = {
                'n_outliers': int((predictions == -1).sum()),
                'notes': {
                    'predictions': predictions.tolist(),
                    'outlier_scores': scores.tolist(),
                    'outlier_indices': data_1.index[predictions == -1].tolist()
                }
            }
            return result

if __name__ == "__main__":
    # Example usage
    prior_config = Dataset(dataset_name="blood-transfusion-service-center",
                           mode="minio")

    prior = prior_config.fetch_prior_dataset()

    evaluator = IsolationForestEvaluator(notes=True)
    results = evaluator.evaluate(prior)
    print(results)