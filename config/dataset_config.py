import os
import shutil
import subprocess
import sys

import pandas as pd


class DatasetConfig:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_path = "../data/{dataset_name}.csv".format(dataset_name=dataset_name)

    def get_config(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path
        }

    def fetch_from_local(self) -> pd.DataFrame:
        import os
        import pandas as pd

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.dataset_path}")

        return pd.read_csv(self.dataset_path)

    def curate_dataset(self) -> None:
        """
        NOT WORKING YET
        Curates a dataset by running its creation script and moving the output.

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