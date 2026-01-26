from synqtab.experiments.Experiment import Experiment


class RebalancingExperiment(Experiment):

    @classmethod
    def short_name(cls):
        from synqtab.enums import ExperimentType
        return str(ExperimentType.REBALANCING)
    
    def _run(self):
        return super()._run()