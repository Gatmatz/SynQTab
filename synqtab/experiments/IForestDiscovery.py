from pathlib import Path
from synqtab.datasets.Dataset import Dataset
from synqtab.evaluators.IsolationForestEvaluator import IsolationForestEvaluator

from synqtab.utils.logging_utils import get_logger
from synqtab.utils.minio_utils import upload_json_to_bucket, MinioBucket, MinioFolder

logger = get_logger(__name__)

class IForestDiscovery:
    def __init__(self):
        self.evaluator = IsolationForestEvaluator()

    def evaluate(self, dataset_name: str) -> dict:
        data_config = Dataset(dataset_name, mode="minio")
        try:
            dataset = data_config.fetch_prior_dataset()
        except Exception:
            logger.exception("Failed to load dataset=%s", dataset_name)
            raise

        result_json = self.evaluator.evaluate(dataset)
        logger.info("Completed Isolation Forest outlier detection for dataset=%s", dataset_name)
        if result_json is None:
            logger.warning("No result returned for dataset=%s", dataset_name)
        else:
            upload_json_to_bucket(
                result_json,
                bucket_name=MinioBucket.EVALUATION,
                folder=f"outlier_detection_iforest/{MinioFolder.PERFECT.value}",
                file_name=f"{dataset_name}_iforest.json"
            )



list_path = Path(__file__).resolve().parent.parent / 'tabarena_list.txt'

iforest = IForestDiscovery()

if not list_path.exists():
    logger.error("Dataset list file not found: %s", list_path)
else:
    with list_path.open() as fh:
        for raw in fh:
            dataset_name = raw.strip()
            if not dataset_name or dataset_name.startswith("#"):
                continue
            try:
                logger.info("Running Isolation Forest outlier detection for dataset=%s", dataset_name)
                iforest.evaluate(dataset_name)
            except Exception:
                logger.exception("Pipeline failed for dataset=%s; continuing", dataset_name)