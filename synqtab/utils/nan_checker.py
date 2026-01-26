"""
Utility script to detect and report NaN values in datasets.

This script helps identify datasets containing NaN (Not a Number) values,
which can cause issues with ML models and evaluators like LOF that require
finite numeric values.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.Dataset import Dataset
from utils.db_utils import get_logger

logger = get_logger(__name__)


class NaNChecker:
    """Utility class to check for NaN values in datasets."""

    def __init__(self, mode: str = "minio"):
        """
        Initialize the NaN checker.

        Args:
            mode: The mode to use for loading datasets ("minio", "db", or "local")
        """
        self.mode = mode

    def check_dataset(self, dataset_name: str, max_rows: Optional[int] = None) -> Dict:
        """
        Check a single dataset for NaN values.

        Args:
            dataset_name: Name of the dataset to check
            max_rows: Maximum number of rows to load (None for all rows)

        Returns:
            Dictionary containing NaN statistics for the dataset
        """
        try:
            data_config = Dataset(dataset_name, mode=self.mode)
            df = data_config.fetch_prior_dataset(max_rows=max_rows)

            total_rows = len(df)
            total_cells = df.size

            # Check for NaN values
            nan_mask = df.isna()
            total_nans = nan_mask.sum().sum()

            if total_nans == 0:
                return {
                    "dataset_name": dataset_name,
                    "has_nans": False,
                    "total_rows": total_rows,
                    "total_columns": len(df.columns),
                    "total_cells": total_cells,
                    "total_nans": 0,
                    "nan_percentage": 0.0,
                    "columns_with_nans": {}
                }

            # Get detailed information about NaN values per column
            columns_with_nans = {}
            for col in df.columns:
                col_nans = nan_mask[col].sum()
                if col_nans > 0:
                    columns_with_nans[col] = {
                        "nan_count": int(col_nans),
                        "nan_percentage": float(col_nans / total_rows * 100),
                        "dtype": str(df[col].dtype)
                    }

            nan_percentage = (total_nans / total_cells) * 100

            return {
                "dataset_name": dataset_name,
                "has_nans": True,
                "total_rows": total_rows,
                "total_columns": len(df.columns),
                "total_cells": total_cells,
                "total_nans": int(total_nans),
                "nan_percentage": float(nan_percentage),
                "columns_with_nans": columns_with_nans,
                "rows_with_any_nan": int(nan_mask.any(axis=1).sum())
            }

        except Exception as e:
            logger.error(f"Failed to check dataset {dataset_name}: {e}")
            return {
                "dataset_name": dataset_name,
                "error": str(e)
            }

    def check_multiple_datasets(self, dataset_names: List[str],
                               max_rows: Optional[int] = None) -> List[Dict]:
        """
        Check multiple datasets for NaN values.

        Args:
            dataset_names: List of dataset names to check
            max_rows: Maximum number of rows to load per dataset

        Returns:
            List of dictionaries containing NaN statistics for each dataset
        """
        results = []
        for dataset_name in dataset_names:
            logger.info(f"Checking dataset: {dataset_name}")
            result = self.check_dataset(dataset_name, max_rows)
            results.append(result)

        return results

    def check_from_file(self, list_path: Path, max_rows: Optional[int] = None) -> List[Dict]:
        """
        Check datasets listed in a file.

        Args:
            list_path: Path to file containing dataset names (one per line)
            max_rows: Maximum number of rows to load per dataset

        Returns:
            List of dictionaries containing NaN statistics for each dataset
        """
        if not list_path.exists():
            logger.error(f"Dataset list file not found: {list_path}")
            return []

        dataset_names = []
        with list_path.open() as fh:
            for line in fh:
                dataset_name = line.strip()
                if dataset_name and not dataset_name.startswith("#"):
                    dataset_names.append(dataset_name)

        return self.check_multiple_datasets(dataset_names, max_rows)

    @staticmethod
    def print_report(results: List[Dict], show_all: bool = False, output_file: Optional[Path] = None):
        """
        Print a formatted report of NaN check results.

        Args:
            results: List of NaN check results
            show_all: If True, show all datasets; if False, only show datasets with NaNs
            output_file: Optional path to save the report to a file
        """
        # Build report content as a list of lines
        report_lines = []
        report_lines.append("\n" + "="*80)
        report_lines.append("NaN DETECTION REPORT")
        report_lines.append("="*80 + "\n")

        datasets_with_nans = [r for r in results if r.get("has_nans", False)]
        datasets_without_nans = [r for r in results if not r.get("has_nans", False) and "error" not in r]
        datasets_with_errors = [r for r in results if "error" in r]

        # Summary
        report_lines.append(f"Total datasets checked: {len(results)}")
        report_lines.append(f"Datasets with NaN values: {len(datasets_with_nans)}")
        report_lines.append(f"Datasets without NaN values: {len(datasets_without_nans)}")
        report_lines.append(f"Datasets with errors: {len(datasets_with_errors)}")
        report_lines.append("\n" + "-"*80 + "\n")

        # Datasets with NaNs (always shown)
        if datasets_with_nans:
            report_lines.append("DATASETS WITH NaN VALUES:")
            report_lines.append("-"*80)
            for result in datasets_with_nans:
                report_lines.append(f"\n📊 Dataset: {result['dataset_name']}")
                report_lines.append(f"   Total rows: {result['total_rows']:,}")
                report_lines.append(f"   Total columns: {result['total_columns']}")
                report_lines.append(f"   Total NaN values: {result['total_nans']:,} ({result['nan_percentage']:.2f}% of all cells)")
                report_lines.append(f"   Rows with any NaN: {result.get('rows_with_any_nan', 'N/A'):,}")
                report_lines.append(f"\n   Columns with NaN values:")

                for col, info in result['columns_with_nans'].items():
                    report_lines.append(f"      • {col}: {info['nan_count']} NaNs ({info['nan_percentage']:.2f}%) [dtype: {info['dtype']}]")
                report_lines.append("")

        # Datasets with errors
        if datasets_with_errors:
            report_lines.append("\n" + "-"*80)
            report_lines.append("DATASETS WITH ERRORS:")
            report_lines.append("-"*80)
            for result in datasets_with_errors:
                report_lines.append(f"   ❌ {result['dataset_name']}: {result['error']}")
            report_lines.append("")

        # Datasets without NaNs (only if show_all is True)
        if show_all and datasets_without_nans:
            report_lines.append("\n" + "-"*80)
            report_lines.append("DATASETS WITHOUT NaN VALUES:")
            report_lines.append("-"*80)
            for result in datasets_without_nans:
                report_lines.append(f"   ✓ {result['dataset_name']} ({result['total_rows']:,} rows, {result['total_columns']} columns)")
            report_lines.append("")

        report_lines.append("="*80 + "\n")

        # Join all lines into a single string
        report_text = "\n".join(report_lines)

        # Print to console
        print(report_text)

        # Optionally save to file
        if output_file:
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open('w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report to file: {e}")


def main():
    """Main entry point for the NaN checker script."""
    parser = argparse.ArgumentParser(
        description="Check datasets for NaN values that can cause issues with ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single dataset
  python utils/nan_checker.py --dataset Marketing_Campaign
  
  # Check all datasets in tabarena_list.txt
  python utils/nan_checker.py --file tabarena_list.txt
  
  # Check with a different mode (db or local)
  python utils/nan_checker.py --dataset Marketing_Campaign --mode db
  
  # Show all datasets (including those without NaNs)
  python utils/nan_checker.py --file tabarena_list.txt --show-all
  
  # Check only first 1000 rows
  python utils/nan_checker.py --dataset Marketing_Campaign --max-rows 1000
  
  # Save report to a file
  python utils/nan_checker.py --file tabarena_list.txt --output reports/nan_report.txt
        """
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Name of a single dataset to check"
    )

    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to file containing dataset names (one per line)"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="minio",
        choices=["minio", "db", "local"],
        help="Mode for loading datasets (default: minio)"
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to load per dataset (default: all rows)"
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all datasets in report, not just those with NaNs"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to save the report to a .txt file"
    )

    args = parser.parse_args()

    if not args.dataset and not args.file:
        parser.error("Either --dataset or --file must be specified")

    checker = NaNChecker(mode=args.mode)

    if args.dataset:
        results = [checker.check_dataset(args.dataset, max_rows=args.max_rows)]
    elif args.file:
        results = checker.check_from_file(args.file, max_rows=args.max_rows)
    else:
        results = []

    # Print report
    NaNChecker.print_report(results, show_all=args.show_all, output_file=args.output)

    # Exit with error code if any datasets have NaNs
    datasets_with_nans = [r for r in results if r.get("has_nans", False)]
    if datasets_with_nans:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

