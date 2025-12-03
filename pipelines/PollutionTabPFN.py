from datasets.Dataset import Dataset
from generators import TabPFN
from utils.utils import write_dataframe_to_db

class PollutionTabPFN:
    def __init__(self, model_settings:dict):
        self.model_settings = model_settings

    def run(self, dataset_name):
        # === Load data from CSV ===
        data_config = Dataset(self.dataset_name, max_rows=1000)
        X, y = data_config.get_dataset()

        # === Generate synthetic data with TabPFN ===
        generator = TabPFN(settings=self.settings)
        X_synth, y_synth = generator.generate(X, y, task=data_config.problem_type)

        # === Save generated data to DB ===
        df = data_config.concatenate_X_y(X_synth, y_synth)
        print(df)
        write_dataframe_to_db(df, table_name='clean_tabpfn_amazon_employee_access', if_exists='replace')


if __name__ == "__main__":
    dataset_name = 'Amazon_employee_access'
    tabpfg_settings = {
        'n_sgld_steps': 10,
        'n_samples': 1,
        'balance_classes': False
    }
    pipeline = PollutionTabPFN(dataset_name, model_settings=tabpfg_settings)
    pipeline.run()