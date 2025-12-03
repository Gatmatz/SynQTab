from evaluators import Evaluator


class Syntheval(Evaluator):
    def __init__(self, clean_X, clean_y, dirty_X, dirty_y, settings: dict):
        """
        Initialize the Syntheval evaluator with synthetic datasets coming from clean and dirty datasets.
        :param clean_X: Synthetic features from clean dataset
        :param clean_y: Synthetic labels from clean dataset
        :param dirty_X: Synthetic features from dirty dataset
        :param dirty_y: Synthetic labels from dirty dataset
        """
        super().__init__(clean_X, clean_y, dirty_X, dirty_y)
        self.settings = settings

    def evaluate(self) -> dict:
        pass