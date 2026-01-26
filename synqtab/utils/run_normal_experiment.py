import warnings
warnings.filterwarnings("ignore") # mitigates synthcity's annoying verbosity


from synqtab.data import Dataset
from synqtab.enums import DataPerfectness
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
                # normal_experiment.run()
                id_before = str(normal_experiment)
                experiment, seed = Experiment.from_str(str(normal_experiment))
                id_after = str(experiment)
                assert id_before == id_after, "Den einai idia"
                # exit(0)
            except Exception as e:
                # import traceback
                
                LOG.error(
                    f'The experiment {str(normal_experiment)} failed but I will continue to the next one.' +
                    f'Error: {e}.',
                    extra={'experiment_id': str(normal_experiment)}
                )
                exit(1)
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
                            normal_experiment = NormalExperiment(
                                dataset=dataset,
                                generator=model,
                                data_error_type=error,
                                data_error_rate=error_rate,
                                data_perfectness=perfectness_level,
                                evaluation_methods=experimental_params.get('evaluation_methods'),
                            )
                            # normal_experiment.run().populate_tasks()
                            id_before = str(normal_experiment)
                            experiment, seed = Experiment.from_str(str(normal_experiment))
                            id_after = str(experiment)
                            assert id_before == id_after, "Den einai idia"
                            # print(normal_experiment)
                            
                        except Exception as e:
                            LOG.error(
                                f"The experiment failed but I will continue to the next one. Error: {e}",
                                extra={'experiment_id': str(normal_experiment)}
                            )
                            continue
