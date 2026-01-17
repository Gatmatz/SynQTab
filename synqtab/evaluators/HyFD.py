from pathlib import Path
import subprocess
import json
import os

from synqtab.evaluators.Evaluator import Evaluator


class HyFD(Evaluator):
    """ HYFD Functional Dependency Discovery Evaluator. Leverages
    https://hpi.de/naumann/projects/data-profiling-and-analytics/metanome-data-profiling/algorithms.html. Parameters:
        - [*required*] `'data'`: the data to perform FD discovery on
        - [*optional*] `'notes'`: True/False on whether to include notes in the result or not.
        If absent, defaults to False.
    """
    
    def compute_result(self):
        data = self.params.get('data')
        temp_csv_path = "temp_data.csv"
        data.to_csv(temp_csv_path, index=False)
        
        try:
            # Run HyFD on the temporary CSV file
            self.run_hyfd(data_path=temp_csv_path)

            # Parse the results and return simplified JSON format
            results = self.parse_hyfd_results()

            if self.params.get('notes', False):
                return results["num_fds"], {'FDs': results['fds']}
            return results["num_fds"]
            
        finally:
            # Clean up the temporary CSV file
            Path(temp_csv_path).unlink(missing_ok=True)

    def run_hyfd(self, data_path: str):
        # Call the Java executable directly from Python
        cmd = [
            "java", "-Xmx16g", "-cp", "../jars/metanome-cli.jar:../jars/HyFD.jar",
            "de.metanome.cli.App",
            "--algorithm", "de.metanome.algorithms.hyfd.HyFD",
            "--file-key", "INPUT_GENERATOR",
            "--files", data_path,
            "--separator", ",",  # Specify comma separator as a direct option
            "--header",  # Indicate that first row is a header
            "--output", "file"
        ]
        # Redirect stdout and stderr to suppress terminal output
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def parse_hyfd_results(self) -> dict:
        """
        Parse the HyFD results file and return a simplified JSON format.

        Returns:
            dict: A dictionary containing:
                - 'num_fds': The number of functional dependencies found
                - 'fds': A list of FDs in the format "A -> B"
        """
        # Find the most recent results file in the results directory
        results_dir = Path("results")
        if not results_dir.exists():
            return {"num_fds": 0, "fds": []}

        # Get all files ending with _fds in the results directory
        result_files = sorted(results_dir.glob("*_fds"), key=os.path.getmtime, reverse=True)

        if not result_files:
            return {"num_fds": 0, "fds": []}

        # Read the most recent results file
        results_file = result_files[0]
        fds = []

        try:
            with open(results_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        fd_data = json.loads(line)
                        if fd_data.get("type") == "FunctionalDependency":
                            # Extract determinant columns
                            determinant_cols = [
                                col["columnIdentifier"]
                                for col in fd_data["determinant"]["columnIdentifiers"]
                            ]

                            # Extract dependant column
                            dependant_col = fd_data["dependant"]["columnIdentifier"]

                            # Format as "A,B -> C" for multiple determinants or "A -> B" for single
                            determinant_str = ",".join(determinant_cols)
                            fd_str = f"{determinant_str} -> {dependant_col}"
                            fds.append(fd_str)
                    except json.JSONDecodeError:
                        continue
        finally:
            # Delete the results file after parsing
            results_file.unlink(missing_ok=True)

        return {
            "num_fds": len(fds),
            "fds": fds
        }
