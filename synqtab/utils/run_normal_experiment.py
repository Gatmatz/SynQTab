import warnings
warnings.filterwarnings("ignore") # mitigates synthcity's annoying verbosity


from synqtab.data import Dataset
from synqtab.enums import DataPerfectness, DataErrorType, ProblemType
from synqtab.experiments.Experiment import Experiment
from synqtab.experiments import NormalExperiment
from synqtab.reproducibility import ReproducibleOperations
from synqtab.utils import get_logger, get_experimental_params_for_normal


LOG = get_logger(__file__)


experimental_params = get_experimental_params_for_normal()

# First, generate all perfect synthetic data (S)
for random_seed in experimental_params.get('random_seeds'):
    ReproducibleOperations.set_random_seed(random_seed)
    for dataset_name in experimental_params.get('dataset_names'):
        dataset = Dataset(dataset_name)
        for model in experimental_params.get('models'):
            try:
                normal_experiment = NormalExperiment(
                    dataset=dataset,
                    generator=model,
                    data_error_type=None,
                    data_error_rate=None,
                    data_perfectness=DataPerfectness.PERFECT, # only perfect data at first
                    evaluation_methods=None,
                )

                normal_experiment.run() # force-compute the regression datasets
            except Exception as e:
                LOG.error(
                    f'The experiment {str(normal_experiment)} failed but I will continue to the next one.' +
                    f'Error: {e}.',
                    extra={'experiment_id': str(normal_experiment)}
                )
                # exit(0)
                continue

# exit(0)

# Then, generate all imperfect (S_hat) and semi-perfect (S_semi) and populate evaluation tasks
for random_seed in experimental_params.get('random_seeds'):
    ReproducibleOperations.set_random_seed(random_seed)
    for dataset_name in experimental_params.get('dataset_names'):
        dataset = Dataset(dataset_name)
        for model in experimental_params.get('models'):
            for error in experimental_params.get('error_types'):
                for error_rate in experimental_params.get('error_rates'):
                    for perfectness_level in experimental_params.get('data_perfectness_levels'):
                        try:
                            if perfectness_level == DataPerfectness.SEMIPERFECT and error_rate != 0.4:
                                # We investigate the cleaning dilemma only for 0.4 error rate
                                continue

                            if perfectness_level == DataPerfectness.SEMIPERFECT and error == DataErrorType.NEAR_DUPLICATE:
                                # Semi-perfect for near duplicates is the same as perfect, no need to compute
                                continue
                            
                            normal_experiment = NormalExperiment(
                                dataset=dataset,
                                generator=model,
                                data_error_type=error,
                                data_error_rate=error_rate,
                                data_perfectness=perfectness_level,
                                evaluation_methods=experimental_params.get('evaluation_methods'),
                            )
                            normal_experiment.run().publish_tasks()
                            
                        except Exception as e:
                            LOG.error(
                                f"The experiment failed but I will continue to the next one. Error: {e}",
                                extra={'experiment_id': str(normal_experiment)}
                            )
                            continue
