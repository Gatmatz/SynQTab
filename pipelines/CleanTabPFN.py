from typing import Optional, Dict, Any

from configs.TabPFNSettings import TabPFNSettings
from datasets.Dataset import Dataset
from generators import TabPFN
from pipelines.Pipeline import Pipeline
from utils.utils import write_dataframe_to_db, get_logger

logger = get_logger(__name__)


class CleanTabPFN(Pipeline):
    """
    Pipeline to load a dataset, generate synthetic data with TabPFN,
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

        # Generate synthetic data
        try:
            generator = TabPFN(settings=self.model_settings)
            X_synth, y_synth = generator.generate(X, y, task=data_config.problem_type)
        except Exception:
            logger.exception("TabPFN generation failed for dataset=%s", dataset_name)
            raise

        # Concatenate and write to DB
        df = data_config.concatenate_X_y(X_synth, y_synth)
        table_name = self._default_table_name(dataset_name)
        try:
            write_dataframe_to_db(df, table_name=table_name, if_exists="replace")
            logger.info("Wrote synthetic data to table=%s", table_name)
        except Exception:
            logger.exception("Failed to write dataframe to DB for table=%s", table_name)
            raise

if __name__ == "__main__":
    settings = TabPFNSettings(n_sgld_steps=1,
                              n_samples=1000,
                              balance_classes=False)

    pipeline = CleanTabPFN(model_settings=settings)
    pipeline.run(dataset_name="Amazon_employee_access", max_rows=1000)