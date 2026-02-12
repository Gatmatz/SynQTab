import warnings
warnings.filterwarnings("ignore") # mitigates synthcity's annoying verbosity


from synqtab.enums import DataPerfectness
from synqtab.experiments import MultiExperiment
from synqtab.reproducibility import ReproducibleOperations
from synqtab.utils import get_logger, get_experimental_params_for_multi


LOG = get_logger(__file__)


experimental_params = get_experimental_params_for_multi()
# First, generate all perfect synthetic data (S)
# for random_seed in experimental_params.get('random_seeds'):
#     ReproducibleOperations.set_random_seed(random_seed)
#     for dataset_name in experimental_params.get('dataset_names'):
#         for model in experimental_params.get('models'):
#             try:
#                 multi_experiment = MultiExperiment(
#                     dataset=dataset_name,
#                     generator=model,
#                     drop_unknown_references = True,
#                     data_error_type=None,
#                     data_error_rate=None,
#                     data_perfectness=DataPerfectness.PERFECT, # only perfect data at first
#                     evaluation_methods=None,
#                 )
#                 multi_experiment._run()
#             except Exception as e:
#                 LOG.error(
#                     f'The experiment {str(multi_experiment)} failed but I will continue to the next one.' +
#                     f'Error: {e}.',
#                     extra={'experiment_id': str(multi_experiment)}
#                 )
#                 continue

# Then, generate all imperfect (S_hat) and semi-perfect (S_semi) and populate evaluation tasks
for random_seed in experimental_params.get('random_seeds'):
    ReproducibleOperations.set_random_seed(random_seed)
    for dataset_name in experimental_params.get('dataset_names'):
        for model in experimental_params.get('models'):
            for error in experimental_params.get('error_types'):
                for error_rate in experimental_params.get('error_rates'):
                    for perfectness_level in experimental_params.get('data_perfectness_levels'):
                        try:
                            if perfectness_level == DataPerfectness.SEMIPERFECT and error_rate != 0.4:
                                # We investigate the cleaning dilemma only for 0.4 error rate
                                continue

                            multi_experiment = MultiExperiment(
                                dataset=dataset_name,
                                generator=model,
                                drop_unknown_references = False,
                                data_error_type=error,
                                data_error_rate=error_rate,
                                data_perfectness=perfectness_level,
                                evaluation_methods=None,
                            )
                            multi_experiment._run()                            
                        except Exception as e:
                            LOG.error(
                                f"The experiment failed but I will continue to the next one. Error: {e}",
                                extra={'experiment_id': str(multi_experiment)}
                            )
                            continue
