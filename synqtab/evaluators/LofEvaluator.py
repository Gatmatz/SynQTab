import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from synqtab.datasets import Dataset
from synqtab.evaluators.SingleEvaluator import SingleEvaluator


class LofEvaluator(SingleEvaluator):
    def __init__(self, n_neighbors=5, contamination='auto', notes: bool = True ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.notes = notes

    def evaluate(self, data_1: pd.DataFrame) -> dict:
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric='euclidean'
        )

        # Fit and predict (-1 for outliers, 1 for inliers)
        predictions = lof.fit_predict(data_1)

        # Get the negative outlier factor scores
        scores = lof.negative_outlier_factor_

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

    evaluator = LofEvaluator(notes=True)
    results = evaluator.evaluate(prior)
    print(results)