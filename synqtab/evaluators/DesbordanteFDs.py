import os
import pandas as pd
import desbordante as db

from synqtab.datasets import Dataset
from synqtab.evaluators.SingleEvaluator import SingleEvaluator
from synqtab.utils.db_utils import get_logger

logger = get_logger(__name__)

class DesbordanteFDs(SingleEvaluator):
    def __init__(self, notes: bool = False):
        self.notes = notes

    def evaluate(self, data_1: pd.DataFrame) -> dict:
        try:
            if len(data_1.columns) > 100:
                return None

            # Load data from pandas DataFrame
            pyro_alg = db.fd.algorithms.Default()
            pyro_alg.load_data(table=data_1)

            logger.info("Data loaded into FD discovery algorithm.")

            pyro_alg.execute()
            logger.info("Executed FD discovery algorithm.")

            # Collect functional dependencies
            fds = [str(fd) for fd in pyro_alg.get_fds()]

            if self.notes is False:
                return {
                    "count": len(fds)
                }
            else:
                return {
                    "count": len(fds),
                    "notes": {
                        "functional_dependencies": fds
                    }
                }

        finally:
            # Remove log file if it exists
            if os.path.exists('myeasylog.log'):
                os.remove('myeasylog.log')

if __name__ == "__main__":
    # Example usage
    prior_config = Dataset(dataset_name="blood-transfusion-service-center",
                           mode="minio")

    prior = prior_config.fetch_prior_dataset()

    evaluator = DesbordanteFDs(notes=True)
    results = evaluator.evaluate(prior)
    print(results)