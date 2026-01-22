from synqtab.enums.data import DataErrorType
from synqtab.errors.CategoricalShift import CategoricalShift
from synqtab.errors.GaussianNoise import GaussianNoise
from synqtab.errors.Placeholder import Placeholder
from synqtab.errors.NearDuplicateRow import NearDuplicateRow
from synqtab.errors.Outlier import Outliers
from synqtab.errors.Inconsistency import Inconsistency

DATA_ERROR_TYPE_TO_DATA_ERROR_CLASS = {
    DataErrorType.CATEGORICAL_SHIFT: CategoricalShift,
    DataErrorType.GAUSSIAN_NOISE: GaussianNoise,
    DataErrorType.INCONSISTENCY: Inconsistency,
    DataErrorType.LABEL_ERROR: None, # TODO IMPLEMENT LABEL ERROR
    DataErrorType.NEAR_DUPLICATE: NearDuplicateRow,
    DataErrorType.OUTLIER: Outliers,
    DataErrorType.PLACEHOLDER: Placeholder,
}


from synqtab.enums.experiments import ExperimentType
from synqtab.experiments.AugmentationExperiment import AugmentationExperiment
from synqtab.experiments.NormalExperiment import NormalExperiment
from synqtab.experiments.RebalancingExperiment import RebalancingExperiment
from synqtab.experiments.PrivacyExperiment import PrivacyExperiment

EXPERIMENT_TYPE_TO_EXPERIMENT_CLASS = {
    ExperimentType.NORMAL: NormalExperiment,
    ExperimentType.PRIVACY: PrivacyExperiment,
    ExperimentType.AUGMENTATION: AugmentationExperiment,
    ExperimentType.REBALANCING: RebalancingExperiment,
}


from synqtab.enums.generators import GeneratorModel
from synqtab.generators.RealTabTransformer import RealTabTransformer
from synqtab.generators.SynthcityGenerator import SynthcityGenerator
from synqtab.generators.TabEBM import TabEBM
from synqtab.generators.TabPFN import TabPFN

GENERATOR_MODEL_TO_GENERATOR_CLASS = {
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


from synqtab.enums.evaluators import EvaluationMethod
from synqtab.evaluators.DCREvaluator import DCREvaluator
from synqtab.evaluators.DesbordanteFDs import DesbordanteFDs
from synqtab.evaluators.DisclosureProtectionEvaluator import DisclosureProtectionEvaluator
from synqtab.evaluators.HyFD import HyFD
from synqtab.evaluators.IsolationForestEvaluator import IsolationForestEvaluator
from synqtab.evaluators.LofEvaluator import LofEvaluator
from synqtab.evaluators.MLAugmentationPrecision import MLAugmentationPrecision
from synqtab.evaluators.MLAugmentationRecall import MLAugmentationRecall
from synqtab.evaluators.MLAugmentationRegression import MLAugmentationRegression
from synqtab.evaluators.MLEfficacy import MLEfficacy
from synqtab.evaluators.QualityEvaluator import QualityEvaluator

EVALUATION_METHOD_TO_EVALUATION_CLASS_MAPPING = {
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
