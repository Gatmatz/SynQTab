from synqtab.enums.EasilyStringifyableEnum import EasilyStringifyableEnum

# =========== ALL EVALUATORS ===========
class EvaluationMethod(EasilyStringifyableEnum):
    DCR = 'DCR'
    DFD = 'DFD'
    DPR = 'DPR'
    HFD = 'HFD'
    IFO = 'IFO'
    LOF = 'LOF'
    APR = 'APR'
    ARC = 'ARC'
    AR2 = 'AR2'
    EFF = 'EFF'
    QLT = 'QLT'


# =========== CLASSIFICATION BASED ON THE NUMBER OF INPUT TABLES ===========
# ================ Singular evaluators take only one table as input, e.g., Isolation Forest
SINGULAR_EVALUATORS: list[EvaluationMethod] = [ # 
    EvaluationMethod.DFD,
    EvaluationMethod.HFD,
    EvaluationMethod.IFO,
    EvaluationMethod.LOF,
]

# ================ Dual evaluators take only two tables as input, e.g., Distance from Closest Record
DUAL_EVALUATORS: list[EvaluationMethod] = [
    EvaluationMethod.DCR,
    EvaluationMethod.DPR,
    EvaluationMethod.APR,
    EvaluationMethod.ARC,
    EvaluationMethod.AR2,
    EvaluationMethod.EFF,
    EvaluationMethod.QLT,
]

# =========== CLASSIFICATION BASED ON THE SEMANTICS ===========
# ================ Quality Evaluators focus on the quality of the synthetic data
QUALITY_EVALUATORS: list[EvaluationMethod] = [
    EvaluationMethod.DFD,
    EvaluationMethod.HFD,
    EvaluationMethod.IFO,
    EvaluationMethod.LOF,
    EvaluationMethod.QLT,
]

# ================ Privacy Evaluators focus on the privacy preservation of the synthetic data
PRIVACY_EVALUATORS: list[EvaluationMethod] = [
    EvaluationMethod.DCR,
    EvaluationMethod.DPR,
]

# ================ ML-Focused Evaluators focus on the (downstream) utility of the synthetic data
ML_FOCUSED_EVALUATORS: list[EvaluationMethod] = [
    EvaluationMethod.APR,
    EvaluationMethod.ARC,
    EvaluationMethod.AR2,
    EvaluationMethod.EFF,
]