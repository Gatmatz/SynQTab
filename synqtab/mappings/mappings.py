from synqtab.enums import (
    DataErrorType, ExperimentType,
    GeneratorModel, EvaluationMethod, 
)
from synqtab.errors import (
    DataError, CategoricalShift, GaussianNoise, LabelError,
    Placeholder, NearDuplicateRow, Outliers, Inconsistency
)
from synqtab.evaluators import (
    Evaluator, DCREvaluator, DesbordanteFDs,
    DisclosureProtectionEvaluator, HyFD,
    IsolationForestEvaluator, LofEvaluator,
    MLAugmentationPrecision, MLAugmentationRecall,
    MLAugmentationRegression, MLEfficacy, QualityEvaluator,
)
from synqtab.experiments import (
    Experiment, NormalExperiment, PrivacyExperiment,
    AugmentationExperiment, RebalancingExperiment
)
from synqtab.generators import (
    Generator, SynthcityGenerator,
    RealTabTransformer, TabEBM, TabPFN
)


DATA_ERROR_TYPE_TO_DATA_ERROR_CLASS: dict[DataErrorType, DataError.__class__] = {
    DataErrorType.CATEGORICAL_SHIFT: CategoricalShift,
    DataErrorType.GAUSSIAN_NOISE: GaussianNoise,
    DataErrorType.INCONSISTENCY: Inconsistency,
    DataErrorType.LABEL_ERROR: LabelError,
    DataErrorType.NEAR_DUPLICATE: NearDuplicateRow,
    DataErrorType.OUTLIER: Outliers,
    DataErrorType.PLACEHOLDER: Placeholder,
}


EXPERIMENT_TYPE_TO_EXPERIMENT_CLASS: dict[ExperimentType, Experiment.__class__] = {
    ExperimentType.NORMAL: NormalExperiment,
    ExperimentType.PRIVACY: PrivacyExperiment,
    ExperimentType.AUGMENTATION: AugmentationExperiment,
    ExperimentType.REBALANCING: RebalancingExperiment,
}


GENERATOR_MODEL_TO_GENERATOR_INSTANCE: dict[GeneratorModel, Generator] = {
    GeneratorModel.CTGAN: SynthcityGenerator(GeneratorModel.CTGAN),
    GeneratorModel.NFLOW: SynthcityGenerator(GeneratorModel.NFLOW),
    GeneratorModel.RTVAE: SynthcityGenerator(GeneratorModel.RTVAE),
    GeneratorModel.TVAE: SynthcityGenerator(GeneratorModel.TVAE),
    GeneratorModel.DDPM: SynthcityGenerator(GeneratorModel.DDPM),
    GeneratorModel.ARF: SynthcityGenerator(GeneratorModel.ARF),
    GeneratorModel.MARGINAL_DISTRIBUTIONS: SynthcityGenerator(GeneratorModel.MARGINAL_DISTRIBUTIONS),
    GeneratorModel.BAYESIAN_NETWORK: SynthcityGenerator(GeneratorModel.BAYESIAN_NETWORK),
    GeneratorModel.GREAT: SynthcityGenerator(GeneratorModel.GREAT),
    GeneratorModel.REALTABFORMER: RealTabTransformer(),
    GeneratorModel.TABPFN: TabPFN(),
    GeneratorModel.TABEBM: TabEBM(),
    GeneratorModel.ADSGAN: SynthcityGenerator(GeneratorModel.ADSGAN),
    GeneratorModel.PATEGAN: SynthcityGenerator(GeneratorModel.PATEGAN),
    GeneratorModel.AIM: SynthcityGenerator(GeneratorModel.AIM),
    GeneratorModel.DPGAN: SynthcityGenerator(GeneratorModel.DPGAN),
    GeneratorModel.DECAF: SynthcityGenerator(GeneratorModel.DECAF),
    GeneratorModel.PRIVBAYES: SynthcityGenerator(GeneratorModel.PRIVBAYES),
}


EVALUATION_METHOD_TO_EVALUATION_CLASS: dict[EvaluationMethod, Evaluator.__class__] = {
    EvaluationMethod.DCR: DCREvaluator,
    EvaluationMethod.DFD: DesbordanteFDs,
    EvaluationMethod.DPR: DisclosureProtectionEvaluator,
    EvaluationMethod.HFD: HyFD,
    EvaluationMethod.IFO: IsolationForestEvaluator,
    EvaluationMethod.LOF: LofEvaluator,
    EvaluationMethod.APR: MLAugmentationPrecision,
    EvaluationMethod.ARC: MLAugmentationRecall,
    EvaluationMethod.AR2: MLAugmentationRegression,
    EvaluationMethod.EFF: MLEfficacy,
    EvaluationMethod.QLT: QualityEvaluator,
}
