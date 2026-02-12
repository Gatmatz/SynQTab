from enum import Enum
from typing import Self


class MinioBucket(Enum):
    REAL = 'real'
    SYNTHETIC = 'synthetic'
    TASKS = 'tasks'
    MULTI = 'multitable'

class MinioFolder(Enum):
    PERFECT = 'perfect'
    IMPERFECT = 'imperfect'
    DATA = 'data'
    METADATA = 'metadata'
    
    @staticmethod
    def create_path(*folders: list[Self]):
        return '/'.join([folder if type(folder) == str else folder.value
                         for folder in folders])

