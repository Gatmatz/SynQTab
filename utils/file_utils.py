import os
from typing import Any, List
import yaml
from logging_utils import get_logger

LOG = get_logger(__file__)


def read_yaml_file(yaml_path: str) -> Any:
    """Safely reads the yaml file.

    Args:
        yaml_path (str): the yaml file to read

    Returns:
        Any: the parsed yaml file
    """
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata


def read_files_from_directory(directory: str) -> List:
    """Returns all files in the directory as a list. Raises FileNotFound Error if directory cannot be found.

    Args:
        directory (str): the directory to read files from

    Returns:
        List: a list of file paths contained in the provided directory
    """
    return os.listdir(directory)
