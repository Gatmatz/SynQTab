import random

from synqtab.experiments.Experiment import Experiment
from synqtab.reproducibility import ReproducibleOperations
from synqtab.data import PostgresClient
from synqtab.utils import get_experimental_params_for_normal


params = get_experimental_params_for_normal()

with open('./retries.txt', 'r') as file:
    experiment_ids = [line.rstrip() for line in file]
random.shuffle(experiment_ids)

found = not_found = published = 0

for experiment_id in experiment_ids:
    if not PostgresClient.experiment_exists(experiment_id):
        not_found += 1
        continue
    
    found += 1
    experiment, random_seed = Experiment.from_str(experiment_id)
    ReproducibleOperations.set_random_seed(random_seed)
    experiment.evaluators = params["evaluation_methods"]
    experiment.publish_tasks(force=True)
    published += 1
    
    
total = found + not_found
print(f"{found=}, {published=}, {not_found=}, {total=}")
