from synqtab.experiments.Experiment import Experiment
from synqtab.evaluators.Evaluation import Evaluation
from synqtab.reproducibility import ReproducibleOperations


evaluation_id: str = 'LGD#R#SH'
experiment_id: str = 'NOR#hazelnut-spread-contaminant-detection#16840#SEMI#LER#40#tabpfn'
experiment, random_seed = Experiment.from_str(experiment_id)
ReproducibleOperations.set_random_seed(random_seed)

evaluation: Evaluation = Evaluation.from_str_and_experiment(
    evaluation_id=evaluation_id,
    experiment=experiment,
)
evaluation.run(force=True)
