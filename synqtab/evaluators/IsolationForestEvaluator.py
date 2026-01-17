from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations
from synqtab.evaluators.Evaluator import Evaluator


class IsolationForestEvaluator(Evaluator):
    """ Isolation Forest Outlier Detection Evaluator. Leverages
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html. Parameters:
        - [*required*] `'data'`: the data to perform outlier detection on
        - [*optional*] `'n_estimators'`: the number of estimators to use for the Isolation Forest
        model. If absent, defaults to 100. See the original implementation for details.
        - [*optional*] `'contamination'`: the amount of contamination of the dat set. 
        If absent, defaults to 'auto'. See the original implementation for details.
        - [*optional*] `'notes'`: True/False on whether to include notes in the result or not.
        If absent, defaults to False.
    """
        
    def compute_result(self):
        data = self.params.get('data')
        iso_forest = ReproducibleOperations.get_isolation_forest_model(
            n_estimators=self.params.get('n_estimators', 100),
            contamination=self.params.get('contamination', 'auto'),
        )

        # Fit and predict (-1 for outliers, 1 for inliers)
        predictions = iso_forest.fit_predict(data)

        # Get anomaly scores (more negative = more outlier-like)
        scores = iso_forest.score_samples(data)
        nof_outliers = int((predictions == -1).sum())
        
        if self.get('notes', False):
            return nof_outliers, {
                'predictions': predictions.tolist(),
                'outlier_scores': scores.tolist(),
            }
        return nof_outliers
