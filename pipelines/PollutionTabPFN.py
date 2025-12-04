import logging
from typing import Optional, Dict, Any

from configs import PollutionSettings, TabPFNSettings
from datasets.Dataset import Dataset
from generators import TabPFN
from pipelines.Pipeline import Pipeline
from utils.utils import write_dataframe_to_db
from pollution.Polluter import Polluter

logger = logging.getLogger(__name__)


class PollutionTabPFN(Pipeline):
    """
    Pipeline that loads a dataset, applies a corruption/pollution method, generates
    synthetic data with TabPFN, and writes results to a database table.
    """

    _POLLUTER_MAP = {
        "MCAR": Polluter.MCAR,
        "MAR": Polluter.SCAR,
        "CSCAR": Polluter.CSCAR,
    }

    def __init__(self, model_settings: TabPFNSettings, pollution_settings: PollutionSettings):
        super().__init__(model_settings)
        self.pollution_settings = pollution_settings.to_dict()


    def _resolve_polluter(self):
        return self._POLLUTER_MAP[self.pollution_settings["type"]]

    def _default_table_name(self, dataset_name: str) -> str:
        base = dataset_name.lower().replace(" ", "_")
        return f"polluted_{base}_tabpfn_{self.pollution_settings['type'].lower()}_r{int(self.pollution_settings['row_percent'])}_c{int(self.pollution_settings['column_percent'])}"

    def run(self, dataset_name: str, max_rows: Optional[int] = None):
        logger.info("Starting PollutionTabPFN pipeline for dataset=%s max_rows=%s", dataset_name, max_rows)

        # Load dataset
        data_config = Dataset(dataset_name, max_rows=max_rows)
        data = data_config.load_dataset()

        # Validate settings and pick polluter
        try:
            polluter = self._resolve_polluter()
        except Exception as exc:
            logger.exception("Invalid pollution settings")
            raise

        # Corrupt data
        try:
            data_corrupted, corrupted_rows, corrupted_columns = polluter.corrupt(
                data,
                random_seed=self.pollution_settings["random_seed"],
                row_percent=self.pollution_settings["row_percent"],
                column_percent=self.pollution_settings["column_percent"],
            )
            logger.debug("Corrupted rows=%s columns=%s", len(corrupted_rows), len(corrupted_columns))
        except Exception as exc:
            logger.exception("Error while corrupting data")
            raise

        # Prepare inputs for generator
        X_corr, y_corr = data_config.split_x_y(data_corrupted)
        y_corr = data_config.encode_y(y_corr)

        # Generate synthetic data
        try:
            generator = TabPFN(settings=self.model_settings)
            X_synth, y_synth = generator.generate(X_corr, y_corr, task=data_config.problem_type)
        except Exception as exc:
            logger.exception("TabPFN generation failed")
            raise

        # Concatenate and write to DB
        df = data_config.concatenate_X_y(X_synth, y_synth)
        table_name = self._default_table_name(dataset_name)

        try:
            write_dataframe_to_db(df, table_name=table_name, if_exists="replace")
            logger.info("Wrote synthetic data to table=%s", table_name)
        except Exception:
            logger.exception("Failed to write dataframe to DB")
            raise

if __name__ == "__main__":
    model_settings = TabPFNSettings(n_sgld_steps=1,
                              n_samples=1,
                              balance_classes=False)

    pollution_settings = PollutionSettings(type="MCAR",
                                           row_percent=10,
                                           column_percent=10,
                                           random_seed=42)

    pipeline = PollutionTabPFN(model_settings=model_settings,
                               pollution_settings=pollution_settings)

    pipeline.run(dataset_name="Amazon_employee_access", max_rows=1000)