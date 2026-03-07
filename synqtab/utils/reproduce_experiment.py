import random

from synqtab.experiments.Experiment import Experiment
from synqtab.reproducibility import ReproducibleOperations
from synqtab.data import PostgresClient


with open('./retries.txt', 'r') as file:
    experiment_ids = [line.rstrip() for line in file]
random.shuffle(experiment_ids)
    
for experiment_id in experiment_ids:
    print(experiment_id)
    experiment, random_seed = Experiment.from_str(experiment_id)
    ReproducibleOperations.set_random_seed(random_seed)
    
    if not PostgresClient.experiment_exists(experiment_id):
        experiment.run(force=True).publish_tasks(force=True)
    else:
        print("Experiment is already computed so all good!")
