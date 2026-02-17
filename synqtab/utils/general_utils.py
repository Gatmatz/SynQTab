from typing import Any, Callable
from synqtab.enums import GeneratorModel

def timed_computation(
    computation: Callable,
    params: dict[str, Any],
    precision: int = 2
) -> tuple[Any, float]:
    from timeit import default_timer as timer
    
    start = timer()
    result = computation(**params)
    end = timer()
    return result, round(end - start, precision) # tuple: result, execution_time


def get_experimental_params_for_normal() -> dict[str, Any]:
    import copy
    from pprint import pp
    import random

    from synqtab.data import MinioClient
    from synqtab.enums import (
        DataErrorType, DataPerfectness,
        QUALITY_EVALUATORS, ML_FOCUSED_EVALUATORS,
        GENERIC_MODELS, MinioBucket, MinioFolder
    )
    from synqtab.environment import RANDOM_SEEDS, ERROR_RATES
    
    random_seeds = copy.deepcopy(RANDOM_SEEDS); random.shuffle(random_seeds)
    pp(f"{random_seeds=}")

    dataset_names = MinioClient.list_files_in_bucket_by_file_extension(
        bucket_name=MinioBucket.REAL.value,
        file_extension='parquet',
        prefix=MinioFolder.create_prefix(MinioFolder.PERFECT, MinioFolder.DATA),
    ); random.shuffle(dataset_names)
    pp(f"{dataset_names=}", compact=True); print()

    models = copy.deepcopy(GENERIC_MODELS); random.shuffle(models)
    models = [model for model in models if model != GeneratorModel.TABEBM] # temporarily exclude tabebm
    # models = [model for model in models if model != GeneratorModel.TABPFN] # temporarily exclude tabpfn
    models = [model for model in models if model != GeneratorModel.ARF] # ARF only utilizes CPU and wastes quota
    models = [model for model in models if model != GeneratorModel.REALTABFORMER] # realtabformer.realtabformer.REaLTabFormer.sample() argument after ** must be a mapping, not NoneType'
    pp(f"{models=}", compact=True); print()

    error_types = [error for error in DataErrorType]; random.shuffle(error_types)
    pp(f"{error_types=}", compact=True); print()

    error_rates = copy.deepcopy(ERROR_RATES); random.shuffle(error_rates)
    pp(f"{error_rates=}")

    data_perfectness_levels = [DataPerfectness.IMPERFECT, DataPerfectness.SEMIPERFECT]
    random.shuffle(data_perfectness_levels)
    pp(f"{data_perfectness_levels=}", compact=True); print()

    evaluation_methods = copy.deepcopy(QUALITY_EVALUATORS + ML_FOCUSED_EVALUATORS); random.shuffle(evaluation_methods)
    pp(f"{evaluation_methods=}", compact=True); print()
    
    return {
        'random_seeds': random_seeds,
        'dataset_names': dataset_names,
        'models': models,
        'error_types': error_types,
        'error_rates': error_rates,
        'data_perfectness_levels': data_perfectness_levels,
        'evaluation_methods': evaluation_methods,
    }