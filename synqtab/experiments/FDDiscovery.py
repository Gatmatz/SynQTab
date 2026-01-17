from pathlib import Path
from synqtab.datasets.Dataset import Dataset
from synqtab.evaluators.HyFD import HyFD
from synqtab.utils.logging_utils import get_logger
from synqtab.utils.minio_utils import upload_json_to_bucket, MinioBucket, MinioFolder

logger = get_logger(__name__)

class FDDiscovery:
    def __init__(self):
        self.evaluator = HyFD()

    def evaluate(self, dataset_name: str) -> dict:
        data_config = Dataset(dataset_name, mode="minio")
        try:
            dataset = data_config.fetch_prior_dataset(max_rows=100)
        except Exception:
            logger.exception("Failed to load dataset=%s", dataset_name)
            raise

        result_json = self.evaluator.evaluate(dataset)
        logger.info("Completed FD Discovery for dataset=%s", dataset_name)
        if result_json is None:
            logger.warning("No result returned for dataset=%s", dataset_name)
        else:
            upload_json_to_bucket(
                result_json,
                bucket_name=MinioBucket.EVALUATION,
                folder=f"hyfd/{MinioFolder.PERFECT.value}",
                file_name=f"{dataset_name}_fds.json"
            )



list_path = Path(__file__).resolve().parent.parent / 'tabarena_list_small.txt'

fd = FDDiscovery()

if not list_path.exists():
    logger.error("Dataset list file not found: %s", list_path)
else:
    with list_path.open() as fh:
        for raw in fh:
            dataset_name = raw.strip()
            if not dataset_name or dataset_name.startswith("#"):
                continue
            try:
                logger.info("Running FD Discovery for dataset=%s", dataset_name)
                fd.evaluate(dataset_name)
            except Exception:
                logger.exception("Pipeline failed for dataset=%s; continuing", dataset_name)