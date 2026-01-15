from typing import Dict, Any, Optional, override

import pandas as pd
from synqtab.evaluators.DualEvaluator import DualEvaluator
from sdmetrics.single_column import MissingValueSimilarity as SDMissingValueSimilarity

class MissingValueSimilarity(DualEvaluator):
    def __init__(self, notes:bool = False):
        self.notes = notes

    @override
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict[str, Any], holdout_table:Optional[pd.DataFrame]) -> Any:
        """
        Loops through all columns and computes the missing value similarity using sdmetrics.

        Returns:
            Dict[str, float]: Dictionary containing missing value similarity score for each column,
                             plus a 'total_sum' key with the sum of all scores.
        """
        results = {}

        # Loop through all columns in the real data
        for column in real_data.columns:
            if column in synthetic_data.columns:
                # Compute missing value similarity for this column using sdmetrics
                score = SDMissingValueSimilarity.compute(
                    real_data=real_data[column],
                    synthetic_data=synthetic_data[column]
                )
                results[column] = score

        # Add sum of all missing value similarity scores
        results['total_sum'] = sum(results.values())

        return results

if __name__== "__main__":
    # Example usage
    real_df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 2, 3, 4]
    })

    synthetic_df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [1, 2, None, 4]
    })
    evaluator = MissingValueSimilarity()
    result = evaluator.evaluate(real_df, synthetic_df, {}, None)
    print(result)