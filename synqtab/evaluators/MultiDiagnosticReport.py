from synqtab.evaluators.Evaluator import Evaluator


class MultiDiagnosticReport(Evaluator):   
    def short_name(self):
        from synqtab.enums import EvaluationMethod
        return str(EvaluationMethod.DRP)
    
    def full_name(self):
        return "Multi Diagnostic Report"
        
    def compute_result(self):
        from sdmetrics.reports.multi_table import DiagnosticReport
        try:
            # Initialize the SDMetrics Multitable DiagnosticReport
            diagnostic_report = DiagnosticReport()

            # Generate the report
            diagnostic_report.generate(
                real_data = self.params.get('real_training_data'), 
                synthetic_data=self.params.get('synthetic_data'), 
                metadata=self.params.get('metadata'), 
                verbose=False
            )

            if not self.params.get('notes', False):
                return diagnostic_report.get_score().item()

            # Get detailed property scores
            report = dict()
            properties = diagnostic_report.get_properties()
            for _, row in properties.iterrows():
                prop_name = row['Property']
                clean_prop_name = prop_name.replace(' ', '_')
                report[f'{clean_prop_name}_Score'] = row['Score']

                # Get details for this specific property and convert to dict/list for JSON serialization
                details = diagnostic_report.get_details(property_name=prop_name)
                report[f'{clean_prop_name}_Details'] = details.to_dict(orient='records')

            return diagnostic_report.get_score().item(), report # get_score returns np.float64 and Postgres crashes when inserting; using item() to convert to python native type float

        except Exception as e:
            # TODO LOG ERROR
            report['DiagnosticReport_error'] = str(e)
            return 0., report
