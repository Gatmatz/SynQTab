from typing import Any

import pandas as pd
from synqtab.generators import Generator

class TabPFN(Generator):
    """
    TabPFN synthetic data generator using TabPFN Unsupervised Model.
    """
    def __init__(self):
        super().__init__()
        self.generator = None

    def generate(self, X_initial: pd.DataFrame, y_initial: pd.DataFrame, n_samples: int, metadata: dict[str, Any]):
        import torch
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import OrdinalEncoder

        df = pd.concat([X_initial, y_initial], axis=1)
        original_cols = df.columns
        original_dtypes = df.dtypes
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns
        numeric_columns = df.select_dtypes(exclude=['object', 'category', 'bool']).columns

        X_numeric = df[numeric_columns].values

        _categorical_encoded = None
        encoder = None

        if len(categorical_columns) > 0:
            # OrdinalEncoder converts categories to 0, 1, 2...
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_categorical_encoded = encoder.fit_transform(df[categorical_columns])

        # Concatenate numeric and ordinal categorical data
        if X_categorical_encoded is not None:
            X_final = np.hstack([X_numeric, X_categorical_encoded])
        else:
            X_final = X_numeric

        df_tensor = torch.tensor(X_final, dtype=torch.float32)
        synthetic_data = self._generate_synthetic_data(df_tensor, n_samples)

        # Split generated data
        n_num_cols = len(numeric_columns)
        synth_num_part = synthetic_data[:, :n_num_cols]
        synth_cat_part = synthetic_data[:, n_num_cols:]

        # Create Numeric DataFrame
        synth_num_df = pd.DataFrame(synth_num_part, columns=numeric_columns)

        # Reversing Ordinal Encoding
        if encoder:
            # Clip values to valid range for each categorical column before inverse_transform
            synth_cat_clipped = synth_cat_part.copy()
            for i, categories in enumerate(encoder.categories_):
                max_idx = len(categories) - 1
                synth_cat_clipped[:, i] = np.clip(synth_cat_clipped[:, i], 0, max_idx)
            synth_cat_decoded = encoder.inverse_transform(synth_cat_clipped.astype(int))
            synth_cat_df = pd.DataFrame(synth_cat_decoded, columns=categorical_columns)
        else:
            synth_cat_df = pd.DataFrame()

        # Combine and Reorder
        synth_final = pd.concat([synth_num_df, synth_cat_df], axis=1)
        synth_final = synth_final[original_cols]

        # Restore Original Data Types
        for col in original_cols:
            target_type = original_dtypes[col]
            if pd.api.types.is_integer_dtype(target_type):
                synth_final[col] = synth_final[col].round().astype(target_type)
            else:
                synth_final[col] = synth_final[col].astype(target_type)

        return synth_final

    def _generate_synthetic_data(self, data_tensor, n_samples):
        from synqtab.reproducibility import ReproducibleOperations

        self.generator = ReproducibleOperations.get_tabpfn_unsupervised_model()
        self.generator.fit(data_tensor)
        synthetic_tensor = self.generator.generate_synthetic_data(n_samples=n_samples)
        synthetic_data = synthetic_tensor.detach().numpy()
        return synthetic_data
