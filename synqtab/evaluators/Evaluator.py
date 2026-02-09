from abc import ABC, abstractmethod

from synqtab.data import Dataset


class EvaluationResult():
    def __init__(self, result: int | float, notes: dict | None):
        self.result = result
        self.notes = notes


class Evaluator(ABC):
    """
    Abstract base class for all evaluators.
    """
    
    def __init__(self, params: dict):
        self.params = params
    
    @abstractmethod
    def short_name(self) -> str:
        pass
    
    @abstractmethod
    def full_name(self) -> str:
        pass
    
    def is_compatible_with(self, dataset: Dataset) -> bool:
        return True

    def evaluate(self) -> dict:
        self.prepare_evaluation(self.params)
        evaluation_result = EvaluationResult(self.compute_result(self.params))
        return self._standardize_evaluation_result(evaluation_result)
    
    def prepare_evaluation(self) -> None:
        pass
    
    def compute_result(self) -> tuple[int | float, dict | None]:
        """Performs the actual evaluation and returns a tuple.
        The first element of the tuple is required and corresponds to the numeric result.
        The second element of the tuple is optional and corresponds to notes as dictionary.
        If you **DO** wish to return notes, use `return <aNumber>, <aDictWithNotes>`. 
        If you do **DO NOT** wish to return notes, use `return <aNumber>`.

        Args:
            params (dict): the parameters needed for the computations as dictionary.

        Returns:
            tuple: result (numeric, required), notes (optional, dict). **Use `return <aNumber>, None` for empty notes!**
        """
        pass
    
    def _standardize_evaluation_result(self, evaluation_result: EvaluationResult) -> dict:
        from synqtab.enums import EvaluationOutput
        
        if evaluation_result.notes:
            return {
                EvaluationOutput.RESULT: evaluation_result.result,
                EvaluationOutput.NOTES: evaluation_result.notes,
            }
        return {EvaluationOutput.RESULT: evaluation_result.result}
