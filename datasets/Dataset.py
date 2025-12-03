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
    def __init__(self, dataset_name, max_rows: int = None):
        self.dataset_name = dataset_name
        self.max_rows = max_rows

        # Establish project root to construct absolute paths
        self.project_root = Path(__file__).parent.parent.resolve()
        self.dataset_path = self.project_root / "data" / f"{self.dataset_name}.csv"
        self.schema = None

        yaml_info = self._fetch_yaml()
        self.problem_type = yaml_info["problem_type"]
        self.target_feature = yaml_info["target_feature"]
        self.categorical_features = yaml_info["categorical_features"]

    def get_config(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "problem_type": self.problem_type,
            "target_feature": self.target_feature,
            "categorical_features": self.categorical_features
        }

    def get_dataset(self) -> tuple[DataFrame, ndarray]:
        data = self._fetch_original_from_local()
        if self.max_rows is not None:
            data = self._limit_dataset_size(data)
        X, y = self._x_y_split(data)
        self.schema = X.columns
        y_encoded = self._encode_y(y)
        return X, y_encoded

    def _fetch_original_from_local(self) -> pd.DataFrame:
        import os
        import pandas as pd

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.dataset_path}")

        return pd.read_csv(self.dataset_path)

    def _fetch_yaml(self) -> dict:
        """
        Reads settings from the dataset's YAML file.

        Returns:
            A dictionary containing 'problem_type', 'target_feature',
            and 'categorical_features'.
        """
        settings_path = (
                self.project_root
                / "tabarena_dataset_curation"
                / "dataset_creation_scripts"
                / "datasets"
                / self.dataset_name
                / f"{self.dataset_name}.yaml"
        )

        with settings_path.open("r") as f:
            info = yaml.safe_load(f)

        return {
            "problem_type": info.get("problem_type"),
            "target_feature": info.get("target_feature"),
            "categorical_features": info.get("categorical_features", []),
        }

    def _x_y_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Splits the DataFrame into features (X) and target (y) based on the target column.

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

    def _encode_y(self, y):
        y_encoded = pd.factorize(y)[0]
        return y_encoded

    def concatenate_X_y(self, X: pd.DataFrame, y: ndarray) -> pd.DataFrame:
        """
        Concatenates features DataFrame (X) and target array (y) into a single DataFrame.

        Args:
            X (pd.DataFrame): The features DataFrame.
            y (ndarray): The target array.
        Returns:
            pd.DataFrame: The concatenated DataFrame with target column.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.schema)
        y_series = pd.Series(y, name=self.target_feature)
        df = pd.concat([X.reset_index(drop=True), y_series.reset_index(drop=True)], axis=1)
        return df

    def _limit_dataset_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limits the dataset size to a maximum number of rows.
        :param df: The input DataFrame
        :param max_rows: The maximum number of rows to keep
        Returns:
            pd.DataFrame: The limited DataFrame.
        """
        if len(df) > self.max_rows:
            return df.sample(n=self.max_rows, random_state=42).reset_index(drop=True)
        return df

    def _curate_dataset(self) -> None:
        """
        NOT WORKING YET
        Curates a datasets by running its creation script and moving the output.

        """
        # Assuming this script is in 'project_root/config/'
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        base_path = os.path.join(project_root, "tabarena_dataset_curation/dataset_creation_scripts/datasets")
        script_dir = os.path.join(base_path, self.dataset_name)
        script_name = f"{self.dataset_name}.py"
        generated_csv_name = f"{self.dataset_name}.csv"
        destination_dir = os.path.join(project_root, "data")

        if not os.path.isdir(script_dir):
            print(f"Error: Directory not found: '{script_dir}'")
            return

        original_cwd = os.getcwd()
        try:
            os.chdir(script_dir)
            print(f"Changed directory to '{os.getcwd()}'")

            print(f"Running script '{script_name}'...")
            # Set PYTHONPATH to include the project root to resolve module imports
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run([sys.executable, script_name], check=True, env=env)
            print("Script finished successfully.")

            if os.path.exists(generated_csv_name):
                os.makedirs(destination_dir, exist_ok=True)
                shutil.move(generated_csv_name, os.path.join(destination_dir, generated_csv_name))
                print(f"Moved '{generated_csv_name}' to '{destination_dir}'")
            else:
                print(f"Error: Generated file '{generated_csv_name}' not found.")

        except FileNotFoundError:
            print(f"Error: Script '{script_name}' not found in '{script_dir}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing script '{script_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            os.chdir(original_cwd)
            print(f"Returned to original directory '{original_cwd}'")