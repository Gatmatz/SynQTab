from typing import Any
import pandas as pd
import numpy as np

# Assuming the base class exists as per your snippet
from synqtab.generators import Generator

class TabEBM(Generator):
    """
    TabEBM synthetic data generator wrapper.
    Encapsulates the official TabEBM logic which generates samples per-class.
    """
    def __init__(self):
        super().__init__()
        self.generator = None

    def generate(self, X_initial: pd.DataFrame, y_initial: pd.DataFrame, n_samples: int, metadata: dict[str, Any]):
        from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
        from synqtab.reproducibility import ReproducibleOperations
        
        # ---------------------------------------------------------
        # 1. Metadata & Setup
        # ---------------------------------------------------------
        df = pd.concat([X_initial, y_initial], axis=1)
        original_cols = df.columns
        original_dtypes = df.dtypes
        
        # Identify columns
        X_cat_cols = X_initial.select_dtypes(include=['object', 'category', 'bool']).columns
        X_num_cols = X_initial.select_dtypes(exclude=['object', 'category', 'bool']).columns
        y_col_name = y_initial.columns[0]

        # ---------------------------------------------------------
        # 2. Process X (Features) - Ordinal Encoding for Categorical, Keep Numeric as is
        # ---------------------------------------------------------
        X_num = X_initial[X_num_cols].values
        
        X_encoder = None
        X_cat_encoded = None
        
        if len(X_cat_cols) > 0:
            X_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_cat_encoded = X_encoder.fit_transform(X_initial[X_cat_cols])
        
        if X_cat_encoded is not None:
            X_final = np.hstack([X_num, X_cat_encoded])
        else:
            X_final = X_num
            
        # ---------------------------------------------------------
        # 3. Process y (Target) - LabelEncoding
        # ---------------------------------------------------------
        # TabEBM generates "per class" (class_0, class_1). We must map y to integers 0..N
        y_encoder = LabelEncoder()
        y_final = y_encoder.fit_transform(y_initial.iloc[:, 0])
        
        n_classes = len(y_encoder.classes_)

        # ---------------------------------------------------------
        # 4. TabEBM Specific Execution
        # ---------------------------------------------------------
        # Calculate samples per class to approximately match total requested n_samples
        # (Handling integer division remainder by adding to the first class if needed, 
        # or just simple floor division)
        n_per_class = max(1, n_samples // n_classes)

        # Initialize and Fit the internal model
        # Note: We pass the processed numpy arrays
        self.generator = ReproducibleOperations.get_tabebm_model()
        
        # According to the repo snippet, generate returns a dict: {'class_0': [...], ...}
        data_syn_dict = self.generator.generate(X_final, y_final, num_samples=n_per_class)

        # ---------------------------------------------------------
        # 5. Reconstruct Data from Dictionary
        # ---------------------------------------------------------
        X_syn_list = []
        y_syn_list = []

        # Iterate through the dictionary returned by TabEBM
        # Expected keys: "class_0", "class_1", etc.
        for key, X_batch in data_syn_dict.items():
            # 1. Store the X batch
            X_syn_list.append(X_batch)
            
            # 2. Reconstruct the Y batch
            # Extract the integer class index from the key string "class_i"
            class_index = int(key.split('_')[1])
            
            # Create a vector of this class index with length equal to the X batch
            y_batch_indices = np.full(len(X_batch), class_index)
            y_syn_list.append(y_batch_indices)

        # Concatenate all batches
        X_synth_raw = np.concatenate(X_syn_list, axis=0)
        y_synth_indices = np.concatenate(y_syn_list, axis=0)

        # ---------------------------------------------------------
        # 6. Decoding and Restoration
        # ---------------------------------------------------------
        
        # --- Restore X (Features) ---
        n_num_cols = len(X_num_cols)
        
        # A. Numeric
        X_synth_num = X_synth_raw[:, :n_num_cols]
        X_synth_num_df = pd.DataFrame(X_synth_num, columns=X_num_cols)
        
        # B. Categorical (Inverse transform from ordinal encoding)
        X_synth_cat_df = pd.DataFrame()
        if X_encoder:
            n_cat_cols = len(X_cat_cols)
            X_synth_cat_encoded = X_synth_raw[:, n_num_cols:n_num_cols + n_cat_cols]
            
            # Round to nearest integer and clip to valid range
            X_synth_cat_encoded = np.round(X_synth_cat_encoded)
            
            # Inverse transform back to original categorical values
            X_synth_cat_restored = X_encoder.inverse_transform(X_synth_cat_encoded)
            X_synth_cat_df = pd.DataFrame(X_synth_cat_restored, columns=X_cat_cols)

        # --- Restore y (Target) ---
        # Inverse transform the integer indices (0, 1) back to original labels ('Yes', 'No')
        y_restored_values = y_encoder.inverse_transform(y_synth_indices)
        y_synth_df = pd.DataFrame(y_restored_values, columns=[y_col_name])

        # ---------------------------------------------------------
        # 7. Final Assembly
        # ---------------------------------------------------------
        # Combine X parts
        X_restored = pd.concat([X_synth_num_df, X_synth_cat_df], axis=1)
        
        # Combine X and y
        synth_final = pd.concat([X_restored, y_synth_df], axis=1)
        
        # Ensure correct column order
        synth_final = synth_final[original_cols]
        
        # Restore Data Types
        for col in original_cols:
            target_type = original_dtypes[col]
            
            if pd.api.types.is_integer_dtype(target_type):
                synth_final[col] = synth_final[col].round().astype(target_type)
            else:
                synth_final[col] = synth_final[col].astype(target_type)

        return synth_final
