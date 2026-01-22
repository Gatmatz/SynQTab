import warnings
warnings.filterwarnings("ignore") # mitigates synthcity's annoying verbosity

import copy
from pprint import pp
import random

from synqtab.data.clients.MinioClient import MinioClient
from synqtab.data.Dataset import Dataset
from synqtab.enums.data import DataErrorType, DataPerfectness
from synqtab.enums.evaluators import QUALITY_EVALUATORS
from synqtab.enums.generators import GENERIC_MODELS
from synqtab.enums.minio import MinioBucket, MinioFolder
from synqtab.experiments.NormalExperiment import NormalExperiment
from synqtab.environment.experiment import RANDOM_SEEDS
from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations


random_seeds = copy.deepcopy(RANDOM_SEEDS); random.shuffle(random_seeds)
pp(f"{random_seeds=}")

dataset_names = MinioClient.list_files_in_bucket_by_file_extension(
    bucket_name=MinioBucket.REAL.value,
    file_extension='parquet',
    prefix=MinioFolder.create_path(MinioFolder.PERFECT, MinioFolder.DATA),
); random.shuffle(dataset_names)
pp(f"{dataset_names=}", compact=True); print()

models = copy.deepcopy(GENERIC_MODELS); random.shuffle(models)
pp(f"{models=}", compact=True); print()

errors = [error for error in DataErrorType]; random.shuffle(errors)
pp(f"{errors=}", compact=True); print()

data_perfectness_levels = [DataPerfectness.IMPERFECT, DataPerfectness.SEMIPERFECT]
random.shuffle(data_perfectness_levels)
pp(f"{data_perfectness_levels=}", compact=True); print()

evaluators = copy.deepcopy(QUALITY_EVALUATORS); random.shuffle(evaluators)
pp(f"{evaluators=}", compact=True); print()

exit(0)


# First, generate all perfect synthetic data (S)
for random_seed in random_seeds:
    for dataset_name in dataset_names:
        for model in models:
            for error in errors:
                ReproducibleOperations.set_random_seed(random_seed)
                dataset = Dataset(dataset_name)
                normal_experiment = NormalExperiment(
                    dataset=dataset,
                    generator=model,
                    data_error=error,
                    data_perfectness=DataPerfectness.PERFECT, # only perfect data at first
                    evaluators=None,
                )
                normal_experiment.run().persist()


# Then, generate all imperfect (S_hat) and semi-perfect (S_semi) and populate evaluation tasks
for random_seed in random_seeds:
    for dataset_name in dataset_names:
        for model in models:
            for error in errors:
                for perfectness_level in data_perfectness_levels:
                    ReproducibleOperations.set_random_seed(random_seed)
                    dataset = Dataset(dataset_name)
                    normal_experiment = NormalExperiment(
                        dataset=dataset,
                        generator=model,
                        data_error=error,
                        data_perfectness=perfectness_level,
                        evaluators=evaluators,
                    )
                    normal_experiment.run().persist().polulate_tasks()
