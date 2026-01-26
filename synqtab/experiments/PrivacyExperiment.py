from synqtab.experiments.Experiment import Experiment


class PrivacyExperiment(Experiment):

    @classmethod
    def short_name(cls):
        from synqtab.enums import ExperimentType
        return str(ExperimentType.PRIVACY)
    
    def _run(self):
        return super()._run()