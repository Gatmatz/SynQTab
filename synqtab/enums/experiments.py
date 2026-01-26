from synqtab.enums.EasilyStringifyableEnum import EasilyStringifyableEnum


class ExperimentType(EasilyStringifyableEnum):
    NORMAL = 'NOR'
    PRIVACY = 'PRI'
    AUGMENTATION = 'AUG'
    REBALANCING = 'REB'
