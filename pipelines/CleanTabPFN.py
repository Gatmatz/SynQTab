from pathlib import Path
from typing import Optional, Dict, Any

from configs.TabPFNSettings import TabPFNSettings
from datasets.Dataset import Dataset
from pipelines.Pipeline import Pipeline
from utils.utils import write_dataframe_to_db, get_logger
from generators.TabPFN import TabPFN
logger = get_logger(__name__)


class CleanTabPFN(Pipeline):
    """
    Pipeline to load a dataset, generate synthetic data with TabPFN Unsupervised model,
    and write the result to a database table.
    """

    def __init__(self, model_settings: TabPFNSettings):
        super().__init__(model_settings)

    def _default_table_name(self, dataset_name: str) -> str:
        base = dataset_name.lower().replace(" ", "_")
        return f"clean_{base}_tabpfn"


    def run(self, dataset_name: str, max_rows: Optional[int] = None) -> None:
        logger.info("Starting CleanTabPFN pipeline for dataset=%s max_rows=%s", dataset_name, max_rows)

        # Load dataset
        data_config = Dataset(dataset_name, max_rows=max_rows)
        try:
            dataset = data_config.load_dataset()
        except Exception:
            logger.exception("Failed to load dataset=%s", dataset_name)
            raise

        # Prepare X and y
        X, y = data_config.split_x_y(dataset)
        y = data_config.encode_y(y)
        dataset = data_config.concatenate_X_y(X, y)

        # Generate synthetic data
        try:
            generator = TabPFN(settings=self.model_settings)
            synthetic = generator.generate(dataset, data_config.get_categorical_indices())
        except Exception:
            logger.exception("TabPFN generation failed for dataset=%s", dataset_name)
            raise

        df = data_config.convert_to_df(synthetic)
        table_name = self._default_table_name(dataset_name)
        try:
            write_dataframe_to_db(df, table_name=table_name, if_exists="replace", schema='tabpfn_clean')
            logger.info("Wrote synthetic data to table=%s", table_name)
        except Exception:
            logger.exception("Failed to write dataframe to DB for table=%s", table_name)
            raise

if __name__ == "__main__":
    settings = TabPFNSettings(n_samples = 1000)

    pipeline = CleanTabPFN(model_settings=settings)
    list_path = Path(__file__).resolve().parent.parent / 'tabarena_list_a.txt'
    max_rows = None

    if not list_path.exists():
        logger.error("Dataset list file not found: %s", list_path)
    else:
        with list_path.open() as fh:
            for raw in fh:
                dataset_name = raw.strip()
                if not dataset_name or dataset_name.startswith("#"):
                    continue
                try:
                    logger.info("Running pipeline for dataset=%s", dataset_name)
                    pipeline.run(dataset_name=dataset_name, max_rows=max_rows)
                except Exception:
                    logger.exception("Pipeline failed for dataset=%s; continuing", dataset_name)