from typing import Any
import yaml


class SingletonFileSystemClient(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonFileSystemClient, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _FileSystemClient:
    something=None
    

class FileSystemClient(_FileSystemClient, metaclass=SingletonFileSystemClient):
    
    @classmethod
    def read_yaml_file(cls, yaml_path: str) -> Any:
        with open(yaml_path, 'r') as file:
            content = yaml.safe_load(file)
        return content
    
    @classmethod
    def write_yaml_file(cls, content: dict, yaml_path: str) -> None:
        with open(yaml_path, 'w') as file:
            yaml.dump(content, file)
            
    @classmethod
    def read_files_from_directory(cls, directory: str) -> list:
        import os
        return os.listdir(directory)
