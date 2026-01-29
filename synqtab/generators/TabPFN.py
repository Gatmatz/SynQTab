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
        from sklearn.preprocessing import OneHotEncoder
        
        df = pd.concat([X_initial, y_initial], axis=1)
        original_cols = df.columns
        original_dtypes = df.dtypes
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns
        numeric_columns = df.select_dtypes(exclude=['object', 'category', 'bool']).columns
        
        X_numeric = df[numeric_columns].values

        # Handle Categorical: Use OneHotEncoder which stores the categories for reversal
        X_categorical_encoded = None
        encoder = None
        
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_categorical_encoded = encoder.fit_transform(df[categorical_columns])
        
        # Concatenate numeric and encoded categorical data
        if X_categorical_encoded is not None:
            X_final = np.hstack([X_numeric, X_categorical_encoded])
        else:
            X_final = X_numeric

        df_tensor = torch.tensor(X_final, dtype=torch.float32)
        synthetic_data = self._generate_synthetic_data(df_tensor, n_samples)

        n_num_cols = len(numeric_columns)
        synth_num = synthetic_data[:, :n_num_cols]
        
        synth_cat_restored = []
        if encoder:
            # The OHE columns are appended after the numeric columns
            synth_cat_encoded = synthetic_data[:, n_num_cols:]
            
            # We cannot use encoder.inverse_transform directly because the model outputs
            # soft probabilities (floats), not hard 0s and 1s.
            # We must iterate through the category blocks and perform an argmax.
            current_idx = 0
            for i, categories in enumerate(encoder.categories_):
                n_unique = len(categories)
                # Slice the columns corresponding to this specific feature
                feature_slice = synth_cat_encoded[:, current_idx : current_idx + n_unique]
                
                # Find the index of the highest value (most likely category)
                # This turns the "soft" prediction back into a hard category
                predicted_indices = np.argmax(feature_slice, axis=1)
                
                # Map indices back to original labels
                restored_col = categories[predicted_indices]
                synth_cat_restored.append(restored_col)
                
                current_idx += n_unique
                
            # Transpose to get shape (n_samples, n_cat_cols)
            synth_cat_df = pd.DataFrame(np.array(synth_cat_restored).T, columns=categorical_columns)
        else:
            synth_cat_df = pd.DataFrame()

        # Create Numeric DataFrame
        synth_num_df = pd.DataFrame(synth_num, columns=numeric_columns)

        # Combine Categorical and Numeric to final df
        synth_final = pd.concat([synth_num_df, synth_cat_df], axis=1)
        
        # Reorder columns to match original input
        synth_final = synth_final[original_cols]
        
        # Restore Data Types
        for col in original_cols:
            target_type = original_dtypes[col]
            
            if pd.api.types.is_integer_dtype(target_type):
                synth_final[col] = synth_final[col].round().astype(target_type)
                continue
            synth_final[col] = synth_final[col].astype(target_type)

        return synth_final
    
    def _generate_synthetic_data(self, data_tensor, n_samples):
        from synqtab.reproducibility import ReproducibleOperations
        from tabpfn_extensions import TabPFNUnsupervisedModel
        
        classifier = ReproducibleOperations.get_tabpfn_classifier_model()
        regressor = ReproducibleOperations.get_tabpfn_regression_model()
        self.generator = TabPFNUnsupervisedModel(classifier, regressor)
        self.generator.fit(data_tensor)
        synthetic_tensor = self.generator.generate_synthetic_data(n_samples=n_samples)
        synthetic_data = synthetic_tensor.detach().numpy()
