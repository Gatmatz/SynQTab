import warnings
warnings.filterwarnings("ignore") # mitigates synthcity's annoying verbosity


from synqtab.enums import DataPerfectness
from synqtab.experiments import MultiFKExperiment
from synqtab.reproducibility import ReproducibleOperations
from synqtab.utils import get_logger, get_experimental_params_for_multi_fk


LOG = get_logger(__file__)


experimental_params = get_experimental_params_for_multi_fk()
# First, generate all perfect synthetic data (S)
for random_seed in experimental_params.get('random_seeds'):
    ReproducibleOperations.set_random_seed(random_seed)
    for dataset_name in experimental_params.get('dataset_names'):
        for model in experimental_params.get('models'):
            try:
                multi_experiment = MultiFKExperiment(
                    dataset=dataset_name,
                    generator=model,
                    drop_unknown_references = True,
                    data_error_type=None,
                    data_error_rate=None,
                    data_perfectness=DataPerfectness.PERFECT, # only perfect data at first
                    evaluation_methods=None,
                )
                multi_experiment._run()
            except Exception as e:
                LOG.error(
                    f'The experiment {str(multi_experiment)} failed but I will continue to the next one.' +
                    f'Error: {e}.',
                    extra={'experiment_id': str(multi_experiment)}
                )
                continue

# Then, generate all imperfect (S_hat) and populate evaluation tasks
# for random_seed in experimental_params.get('random_seeds'):
#     ReproducibleOperations.set_random_seed(random_seed)
#     for dataset_name in experimental_params.get('dataset_names'):
#         for model in experimental_params.get('models'):
#             for error in experimental_params.get('error_types'):
#                 for error_rate in experimental_params.get('error_rates'):
#                     try:
#                         multi_experiment = MultiFKExperiment(
#                             dataset=dataset_name,
#                             generator=model,
#                             drop_unknown_references = False,
#                             data_error_type=error,
#                             data_error_rate=error_rate,
#                             data_perfectness=DataPerfectness.IMPERFECT,
#                             evaluation_methods=None
#                         )
#                         multi_experiment._run()
#                     except Exception as e:
#                         LOG.error(
#                             f"The experiment failed but I will continue to the next one. Error: {e}",
#                             extra={'experiment_id': str(multi_experiment)}
#                         )
#                         continue
