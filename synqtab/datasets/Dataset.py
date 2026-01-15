import json
from pathlib import Path
import tensorflow as tf
import pandas as pd
import torch
from numpy import ndarray
from synqtab.utils.db_utils import read_table_from_db
from synqtab.utils.minio_utils import read_parquet_from_bucket, MinioBucket, MinioFolder, read_yaml_from_bucket


class Dataset:
    """
    Manages loading and processing of a tabular dataset.

    This class handles fetching dataset information from a YAML file,
    loading the data from a CSV, and providing methods to access
    and manipulate the dataset.

    Attributes:
        dataset_name (str): The name of the dataset.
        max_rows (int, optional): The maximum number of rows to load from the dataset.
        project_root (Path): The root directory of the project.
        dataset_path (Path): The absolute path to the dataset's CSV file.
        schema (pd.Index, optional): The column names of the feature DataFrame (X).
        problem_type (str): The type of machine learning problem (e.g., 'classification').
        target_feature (str): The name of the target column.
        categorical_features (list[str]): A list of names of categorical columns.
    """
    def __init__(self, dataset_name, mode):
        """
        Initializes the Dataset object.

        Args:
            dataset_name (str): The name of the dataset to load.
            max_rows (int, optional): If specified, the dataset will be down-sampled
                to this number of rows. Defaults to None.
        """
        self.dataset_name = dataset_name
        self.max_rows = None
        self.db_schema = None
        self.schema = None
        self.mode = mode

        yaml_info = self._fetch_yaml()
        self.problem_type = yaml_info["problem_type"]
        self.target_feature = yaml_info["target_feature"]
        self.categorical_features = yaml_info["categorical_features"]

    def get_config(self) -> dict:
        """
        Returns the configuration of the dataset.

        Returns:
            dict: A dictionary containing metadata about the dataset.
        """
        return {
            "dataset_name": self.dataset_name,
            "problem_type": self.problem_type,
            "target_feature": self.target_feature,
            "categorical_features": self.categorical_features
        }

    def fetch_prior_dataset(self, max_rows:int = None) -> pd.DataFrame:
        if self.mode == "db":
            dataset = read_table_from_db(self.dataset_name, schema="tabarena_label_encoded")
            if max_rows is not None and len(dataset) > max_rows:
                dataset = dataset.sample(n=max_rows, random_state=42).reset_index(drop=True)

            self.schema = dataset.columns
            return dataset
        elif self.mode == "local":
            dataset = pd.read_csv(f"tabarena/{self.dataset_name}.csv")
            if max_rows is not None and len(dataset) > max_rows:
                dataset = dataset.sample(n=max_rows, random_state=42).reset_index(drop=True)

            self.schema = dataset.columns
            return dataset
        elif self.mode == "minio":
            dataset = read_parquet_from_bucket(
                MinioBucket.REAL,
                f"{MinioFolder.PERFECT.value}/{MinioFolder.DATA.value}/{self.dataset_name}.parquet"
            )
            if max_rows is not None and len(dataset) > max_rows:
                dataset = dataset.sample(n=max_rows, random_state=42).reset_index(drop=True)

            self.schema = dataset.columns
            return dataset

    def fetch_clean_synthetic_dataset(self, model:str = "tabpfn", max_rows: int = None) -> pd.DataFrame:
        dataset = read_table_from_db(f"clean_{self.dataset_name}_{model}", schema=f"{model}_clean")
        if max_rows is not None and len(dataset) > max_rows:
            dataset = dataset.sample(n=max_rows, random_state=42).reset_index(drop=True)

        return dataset

    def _fetch_yaml(self) -> dict:
        """
        Reads settings from the dataset's YAML-style meta table and normalizes types.
        """
        import pandas as pd
        import yaml
        # ------------------ Parsing Helpers ------------------
        def _parse_raw(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            if isinstance(val, (list, tuple, set)):
                return list(val)
            if isinstance(val, str):
                try:
                    return yaml.safe_load(val)
                except Exception:
                    return val
            return val

        def _as_string(val):
            v = _parse_raw(val)
            if v is None:
                return None
            if isinstance(v, list):
                return None if not v else str(v[0])
            return str(v)

        def _as_list(val):
            v = _parse_raw(val)
            if v is None:
                return []
            if isinstance(v, list):
                return v
            if isinstance(v, str):
                return [v]
            return [v]

        if self.mode == "db":
            df = read_table_from_db(f"{self.dataset_name}_meta", schema="tabarena_metadata")

            if df is None or len(df) == 0:
                return {"problem_type": None, "target_feature": None, "categorical_features": []}

            # Convert the two-column table (meta_key, meta_value) → dictionary
            row = dict(zip(df["meta_key"], df["meta_value"]))
        elif self.mode == "minio":
            row = read_yaml_from_bucket(
                bucket_name=MinioBucket.REAL.value,
                object_name=f"{MinioFolder.PERFECT.value}/{MinioFolder.METADATA.value}/{self.dataset_name}.yaml"
            )

            if row is None or len(row) == 0:
                return {"problem_type": None, "target_feature": None, "categorical_features": []}


            return {
                "problem_type": _as_string(row.get("problem_type")),
                "target_feature": _as_string(row.get("target_feature")),
                "categorical_features": _as_list(row.get("categorical_features")),
            }


    def split_x_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Splits the DataFrame into features (X) and target (y) based on the target column.
        This is useful for TabPFN which requires separate X and y inputs.
        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame (X) and the target Series (y).
        """
        if self.target_feature not in df.columns:
            raise ValueError(f"Target column '{self.target_feature}' not found in DataFrame columns.")

        X = df.drop(columns=[self.target_feature])
        y = df[self.target_feature]
        return X, y

    def encode_y(self, y: pd.Series) -> ndarray:
        """
        Encodes the target variable into integer labels.
        This is useful for TabPFN which requires integer labels for classification tasks.
        Args:
            y (pd.Series): The target variable Series.

        Returns:
            ndarray: The encoded target array.
        """
        if self.problem_type == "classification":
            y_encoded = pd.factorize(y)[0]
            return y_encoded
        return y

    def concatenate_X_y(self, X: pd.DataFrame, y: ndarray) -> pd.DataFrame:
        """
        Concatenates features DataFrame (X) and target array (y) into a single DataFrame.
        This is useful when we evaluate the synthetic dataset using Syntheval.
        Syntheval expects a single DataFrame with both features and target.
        Args:
            X (pd.DataFrame): The features DataFrame.
            y (ndarray): The target array.
        Returns:
            pd.DataFrame: The concatenated DataFrame with target column.
        """
        if not isinstance(X, pd.DataFrame):
            columns = self.schema.drop(self.target_feature)
            X = pd.DataFrame(X, columns=columns)

        y_series = pd.Series(y, name=self.target_feature)
        df = pd.concat([X.reset_index(drop=True), y_series.reset_index(drop=True)], axis=1)
        return df

    def convert_to_df(self, table:torch.Tensor) -> pd.DataFrame:
        """
        Converts a TensorFlow tensor to a pandas DataFrame using the stored schema.
        This is useful when the synthetic data is generated as a tensor and needs to be converted back to DataFrame format.
        Args:
            table (tf.tensor): The input TensorFlow tensor.
        Returns:
            pd.DataFrame: The converted DataFrame.
        """
        if isinstance(table, torch.Tensor):
            array = table.cpu().detach().numpy()
        elif isinstance(table, tf.Tensor):
            array = table.numpy()
        else:
            raise ValueError("Input must be a PyTorch or TensorFlow tensor.")

        array = table.numpy()
        df = pd.DataFrame(array, columns=self.schema)
        return df

    def get_categorical_indices(self) -> list[int]:
        """
        Returns the indices of categorical features in the dataset.

        Returns:
            list[int]: A list of indices corresponding to categorical features.
        """
        if self.schema is None:
            raise ValueError("Schema is not defined. Load the dataset first to set the schema.")

        categorical_list = [self.schema.get_loc(col) for col in self.categorical_features if col in self.schema]

        if self.problem_type == "classification":
            target_index = self.schema.get_loc(self.target_feature)
            categorical_list.append(target_index)

        return categorical_list

    def create_sdmetrics_metadata(self) -> dict:
        """
        Creates metadata for sdmetrics based on the dataset configuration.

        Returns:
            dict: A dictionary containing sdmetrics metadata.
        """
        metadata = {
            "columns": {}
        }

        for col in self.schema:
            if col == self.target_feature and self.problem_type == "classification":
                metadata["columns"][col] = {"sdtype": "categorical"}
            elif col in self.categorical_features:
                metadata["columns"][col] = {"sdtype": "categorical"}
            else:
                metadata["columns"][col] = {"sdtype": "numerical"}

        return metadata

    def write_metadata_to_json(self, file_path: str) -> None:
        """
        Writes the sdmetrics metadata to a JSON file.

        Args:
            file_path (str): The path to the JSON file where metadata will be saved.
        """
        metadata = self.create_sdmetrics_metadata()
        metadata_path = Path(f'metadata/{self.dataset_name}.json')
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)