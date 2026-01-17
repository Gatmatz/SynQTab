from sklearn.neighbors import LocalOutlierFactor

from synqtab.evaluators.Evaluator import Evaluator


class LofEvaluator(Evaluator):
    """ Local Outlier Factor (LOF) Outlier Detection Evaluator. Leverages
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html. Parameters:
        - [*required*] `'data'`: the data to perform outlier detection on
        - [*optional*] `'n_neighbors'`: the number of estimators to use for the LOF
        model. If absent, defaults to 5. See the original implementation for details.
        - [*optional*] `'contamination'`: the amount of contamination of the dat set. 
        If absent, defaults to 'auto'. See the original implementation for details.
        - [*optional*] `'notes'`: True/False on whether to include notes in the result or not.
        If absent, defaults to False.
    """
    def __init__(self, n_neighbors=5, contamination='auto', notes: bool = True ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.notes = notes
        
    def compute_result(self):
        data = self.params('data')
        lof = LocalOutlierFactor(
            n_neighbors=self.params.get('n_neighbors', 20),
            contamination=self.params.get('contamination', 'auto'),
            metric='euclidean',
            n_jobs=-1,
        )
        # Fit and predict (-1 for outliers, 1 for inliers)
        predictions = lof.fit_predict(data)

        # Get the negative outlier factor scores
        scores = lof.negative_outlier_factor_
        nof_outliers = int((predictions == -1).sum())
        
        if self.params.get('notes', False):
            return nof_outliers, {
                'predictions': predictions.tolist(),
                'outlier_scores': scores.tolist(),
            }
        return nof_outliers
