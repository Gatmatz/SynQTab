import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from numpy import ndarray
from pandas import DataFrame


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
    def __init__(self, dataset_name, max_rows: int = None):
        """
        Initializes the Dataset object.

        Args:
            dataset_name (str): The name of the dataset to load.
            max_rows (int, optional): If specified, the dataset will be down-sampled
                to this number of rows. Defaults to None.
        """
        self.dataset_name = dataset_name
        self.max_rows = max_rows

        # Establish project root to construct absolute paths
        self.project_root = Path("/kaggle/input/amazon-employee-access/")
        self.dataset_path = self.project_root / f"{self.dataset_name}.csv"
        self.schema = None

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
            "dataset_path": self.dataset_path,
            "problem_type": self.problem_type,
            "target_feature": self.target_feature,
            "categorical_features": self.categorical_features
        }

    def load_dataset(self) -> pd.DataFrame:
        """
        Fetches the original dataset from a local CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            FileNotFoundError: If the dataset CSV file does not exist.
        """
        import os
        import pandas as pd

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.dataset_path}")

        dataset = pd.read_csv(self.dataset_path)

        if self.max_rows is not None and len(dataset) > self.max_rows:
            dataset = dataset.sample(n=self.max_rows, random_state=42).reset_index(drop=True)

        self.schema = dataset.columns
        return dataset

    def _fetch_yaml(self) -> dict:
        """
        Reads settings from the dataset's YAML file.

        Returns:
            A dictionary containing 'problem_type', 'target_feature',
            and 'categorical_features'.
        """
        settings_path = (
                self.project_root
                / f"{self.dataset_name}.yaml"
        )

        with settings_path.open("r") as f:
            info = yaml.safe_load(f)

        return {
            "problem_type": info.get("problem_type"),
            "target_feature": info.get("target_feature"),
            "categorical_features": info.get("categorical_features", []),
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
        y_encoded = pd.factorize(y)[0]
        return y_encoded

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

    def _limit_dataset_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limits the dataset size to a maximum number of rows by random sampling.
        This is useful when working with very large datasets and GPU is limited.
        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame, down-sampled if it was larger than `self.max_rows`.
        """
        if len(df) > self.max_rows:
            return df.sample(n=self.max_rows, random_state=42).reset_index(drop=True)
        return df