from typing import override, Any, Dict

from sdmetrics.reports.single_table import QualityReport
import pandas as pd

from synqtab.datasets import Dataset
from synqtab.evaluators.DualEvaluator import DualEvaluator


class QualityEvaluator(DualEvaluator):
    """
    Evaluator that uses SDMetrics QualityReport to assess the statistical
    quality of data_2 compared to data_1. data_1 can be the real data or the synthetic created from clean data
    and data_2 is the synthetic data or the data created from the polluted data.
    """

    def __init__(self, notes: bool = False):
        self.notes = notes

    @override
    def evaluate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: Dict[str, Any], holdout_table:pd.DataFrame = None) -> Any:
        """
        Evaluates the synthetic data using the SDMetrics QualityReport.
        This generates an aggregate score and detailed property scores for
        Column Shapes and Column Pair Trends.
        """
        report = {}

        try:
            # Initialize the SDMetrics QualityReport
            quality_report = QualityReport()

            # Generate the report
            quality_report.generate(real_data, synthetic_data, metadata=metadata, verbose=False)
            sd_metrics_report = quality_report

            if self.notes is False:
                return {'quality_score': quality_report.get_score()}

            # Get detailed property scores (Column Shapes, Column Pair Trends)
            properties = quality_report.get_properties()
            for _, row in properties.iterrows():
                prop_name = row['Property']
                clean_prop_name = prop_name.replace(' ', '_')
                report[f'{clean_prop_name}_Score'] = row['Score']

                # Get details for this specific property and convert to dict/list for JSON serialization
                details = quality_report.get_details(property_name=prop_name)
                report[f'{clean_prop_name}_Details'] = details.to_dict(orient='records')

            return {
                'quality_score': quality_report.get_score(),
                'notes': report
            }

        except Exception as e:
            report['QualityReport_error'] = str(e)

            return {'quality_score': 0.0, 'notes': report}

if __name__ == "__main__":
    # Example usage
    prior_config = Dataset(dataset_name="blood-transfusion-service-center",
                           mode="minio")

    prior = prior_config.fetch_prior_dataset()
    sd_metadata = prior_config.create_sdmetrics_metadata()

    evaluator = QualityEvaluator(notes=True)
    results = evaluator.evaluate(prior, prior, metadata=sd_metadata)
    print(results)