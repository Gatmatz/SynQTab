from enum import IntEnum
from synqtab.enums import EasilyStringifyableEnum
from synqtab.enums.generators import GeneratorModel


class DataErrorColor(EasilyStringifyableEnum):
    SFT = '#4363d8'  # Royal Blue
    NOI = '#f58231'  # Orange
    PLC = '#911eb4'  # Purple
    DUP = '#42d4f4'  # Sky Blue
    OUT = '#f032e6'  # Magenta
    INC = '#808000'  # Olive
    LER = '#469990'  # Teal/Cyan

class DataErrorMarker(EasilyStringifyableEnum):
    SFT = 'o'  # Circle
    NOI = 's'  # Square
    PLC = 'D'  # Diamond
    DUP = '^'  # Triangle Up
    OUT = 'v'  # Triangle Down
    INC = 'P'  # Plus (filled)
    LER = 'X'  # X (filled)

class PlotFont(EasilyStringifyableEnum):
    FAMILY = 'sans-serif'
    NAME   = 'Ubuntu'        # Round, modern; falls back to Noto Sans then sans-serif
    FALLBACK = 'Noto Sans'

class FontSize(IntEnum):
    X_LABEL   = 18
    Y_LABEL   = 18
    SUBTITLE  = 24
    LEGEND    = 20
    TICK      = 16

MODEL_ORDER = [
    GeneratorModel.CTGAN,
    GeneratorModel.TVAE,
    GeneratorModel.RTVAE,
    GeneratorModel.NFLOW,
    GeneratorModel.GREAT,
    GeneratorModel.TABPFN,
    GeneratorModel.DDPM,
    GeneratorModel.BAYESIAN_NETWORK,
    GeneratorModel.MARGINAL_DISTRIBUTIONS,
    # GeneratorModel.REALTABFORMER,
    # GeneratorModel.TABEBM,
    # GeneratorModel.ARF,
]