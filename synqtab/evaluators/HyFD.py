from pathlib import Path

from synqtab.evaluators.Evaluator import Evaluator
from synqtab.environment import MAX_COLUMNS_FOR_FD_DISCOVERY

# Get absolute path to synqtab package directory (parent of evaluators/)
_SYNQTAB_DIR = Path(__file__).resolve().parent.parent
_JARS_DIR = _SYNQTAB_DIR / "jars"
_JAVA_MEMORY_ALLOCATION_IN_GB = "24"


class HyFD(Evaluator):
    """ HYFD Functional Dependency Discovery Evaluator. Leverages
    https://hpi.de/naumann/projects/data-profiling-and-analytics/metanome-data-profiling/algorithms.html. Parameters:
        - [*required*] `'data'`: the data to perform FD discovery on
        - [*optional*] `'notes'`: True/False on whether to include notes in the result or not.
        If absent, defaults to False.
    """
    
    def short_name(self):
        from synqtab.enums import EvaluationMethod
        return str(EvaluationMethod.HFD)
    
    def full_name(self):
        return "HyFD Functional Dependencies Discovery"
    
    def compute_result(self):
        from synqtab.enums import EvaluationInput
        
        data = self.params.get(str(EvaluationInput.DATA))
        if len(data.columns) >= MAX_COLUMNS_FOR_FD_DISCOVERY:
            return -1
        
        # Use absolute path for temp file in jars directory
        temp_csv_path = _JARS_DIR / "temp_data.csv"
        data.to_csv(temp_csv_path, index=False)
        
        try:
            # Run HyFD on the temporary CSV file
            self.run_hyfd(data_path=str(temp_csv_path))

            # Parse the results and return simplified JSON format
            results = self.parse_hyfd_results()

            notes_enabled = self.params.get('notes', False) or self.params.get(EvaluationInput.NOTES, False)
            if notes_enabled:
                return results["num_fds"], {'FDs': results['fds']}
            return results["num_fds"]
            
        finally:
            # Clean up the temporary CSV file
            temp_csv_path.unlink(missing_ok=True)

    def run_hyfd(self, data_path: str):
        import subprocess
        
        # Build classpath with absolute paths
        metanome_jar = _JARS_DIR / "metanome-cli.jar"
        hyfd_jar = _JARS_DIR / "HyFD.jar"
        classpath = f"{metanome_jar}:{hyfd_jar}"
        
        # Call the Java executable directly from Python
        cmd = [
            "java", f"-Xmx{_JAVA_MEMORY_ALLOCATION_IN_GB}g", "-cp", classpath,
            "de.metanome.cli.App",
            "--algorithm", "de.metanome.algorithms.hyfd.HyFD",
            "--file-key", "INPUT_GENERATOR",
            "--files", data_path,
            "--separator", ",",  # Specify comma separator as a direct option
            "--header",  # Indicate that first row is a header
            "--output", "file"
        ]
        # Run from jars directory so results go there
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=_JARS_DIR)

    def parse_hyfd_results(self) -> dict:
        """
        Parse the HyFD results file and return a simplified JSON format.

        Returns:
            dict: A dictionary containing:
                - 'num_fds': The number of functional dependencies found
                - 'fds': A list of FDs in the format "A -> B"
        """
        import json, os
        
        # Results are created in jars/results/ since we run subprocess with cwd=_JARS_DIR
        results_dir = _JARS_DIR / "results"
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
