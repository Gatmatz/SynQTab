from datasets.Dataset import Dataset
from generators.TabPFN import TabPFN
from generators.Generator import Task

tabpfg_settings = {
    'n_sgld_steps': 10,
    'n_samples': 1,
    'balance_classes': False
}

# === Synthetic Data Generation ===
data_config = Dataset('Amazon_employee_access', max_rows=1000)
X, y = data_config.get_dataset()

generator = TabPFN(settings=tabpfg_settings)

X_synth, y_synth = generator.generate(X, y, task=Task.CLASSIFICATION)

