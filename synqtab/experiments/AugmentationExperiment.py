from synqtab.experiments.Experiment import Experiment
from synqtab.enums.experiments import ExperimentType


class AugmentationExperiment(Experiment):
    
    @classmethod
    def short_name(cls):
        return str(ExperimentType.AUGMENTATION)
    
    def _run(self):
        return super()._run()