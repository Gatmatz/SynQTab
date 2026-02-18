import pandas as pd

def handle_categorical(data:pd.DataFrame, method:str = 'onehot') -> pd.DataFrame:
    """Handles categorical data for outlier detection.
    Methods supported are:
        - 'onehot': applies one-hot encoding to categorical features. This is the default method
        - 'label': applies label encoding to categorical features. This method is not recommended for outlier detection, as it may introduce ordinal relationships where there are none.
        - 'only_numerical': drops all categorical features and only keeps numerical features. This is a simple baseline method that may work well in some cases, but it may also lead to loss of important information.
    Args:
        data (pd.DataFrame): the input data to handle
        method (str, optional): the method to handle categorical data. Defaults to 'onehot'.
    Returns:
        pd.DataFrame: the data with categorical features handled according to the specified method.
    """
    if method == 'onehot':
        from sklearn.preprocessing import OneHotEncoder
        data_copy = data.copy()
        categorical_cols = data_copy.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return data_copy
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cols = encoder.fit_transform(data_copy[categorical_cols])
        encoded_col_names = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=data_copy.index)
        data_copy = data_copy.drop(columns=categorical_cols)
        data_copy = pd.concat([data_copy, encoded_df], axis=1)
        return data_copy
    elif method == 'label':
        from sklearn.preprocessing import OrdinalEncoder
        data_copy = data.copy()
        categorical_cols = data_copy.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            encoder = OrdinalEncoder()
            data_copy[categorical_cols] = encoder.fit_transform(data_copy[categorical_cols])
        return data_copy
    elif method == 'only_numerical':
        return data.select_dtypes(include='number')
    else:
        raise ValueError(f"Unsupported method '{method}' for handling categorical data. Supported methods are: 'onehot', 'label', 'only_numerical'.")