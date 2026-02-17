from typing import Optional, Self
from synqtab.enums.EasilyStringifyableEnum import EasilyStringifyableEnum


class MinioBucket(EasilyStringifyableEnum):
    REAL = 'real'
    SYNTHETIC = 'synthetic'
    TASKS = 'tasks'
    FINISHED_TASKS = 'finished-tasks'
    FAILED_TASKS = 'failed-tasks'
    SKIPPED_TASKS = 'skipped-tasks'

class MinioFolder(EasilyStringifyableEnum):
    PERFECT = 'perfect'
    IMPERFECT = 'imperfect'
    DATA = 'data'
    METADATA = 'metadata'
    
    @staticmethod
    def create_prefix(*folders: list[Self | str], ignore: Optional[Self | str] = None):
        if ignore:
            folders = [folder for folder in folders if folder != ignore]
        return '/'.join([str(folder) for folder in folders])
