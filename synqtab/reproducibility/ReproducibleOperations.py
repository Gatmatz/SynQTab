from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
    
from synqtab.reproducibility import ReproducibilityError


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _RandomSeedOperations:
    _random_seed: int | float = None


class ReproducibleOperations(_RandomSeedOperations, metaclass=Singleton):

    def _ensure_reproducibility(self) -> None:
        if self._random_seed:
            np.random.seed(self._random_seed)
            return
        raise ReproducibilityError(
            f"The reproducibility of the requested operation that involves randomness cannot be ensured. \
                Make sure to call the set_random_seed(some_seed) function at least once in your program."
        )

    @classmethod
    def set_random_seed(self, random_seed: int | float):
        self._random_seed = random_seed

    @classmethod
    def sample_from(
        cls,
        elements: List,
        how_many: int | float,
        at_least: int = 1,
        sampling_with_replacement: bool = False,
    ) -> List:
        """Samples `how_many` items from `elements`. Internally uses numpy. Reproducibility is ensured as long as you stick to
        functions of this class throughout the application for all operations that require randomness.

        Args:
            elements (List): the elements to sample from
            how_many (int | float): the desired number of items to sample
            at_least (int, optional): number of items to be sampled at least. Overrides `how_many` in case `how_many` < `at_least`.
            Defaults to 1.
            sampling_with_replacement (bool, optional): Whether replacement should be applied during sampling. Defaults to False.

        Returns:
            List: the sampled items
        """
        cls._ensure_reproducibility()
        return np.random.choice(
            a=elements,
            size=max(int(how_many), at_least),
            replace=sampling_with_replacement,
        )

    @classmethod
    def uniform(
        cls, low: int | float, high: int | float, size: Optional[int | float] = None
    ):
        """Wraps a call to `np.random.uniform` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.uniform.html.
        """
        cls._ensure_reproducibility()
        return np.random.uniform(low=low, high=high, size=size)

    @classmethod
    def normal(cls, loc: float, scale: float, size: int | Tuple[int]):
        """Wraps a call to `np.random.normal` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.normal.html.
        """
        cls._ensure_reproducibility()
        return np.random.normal(loc=loc, scale=scale, size=size)
    
    @classmethod
    def permutation(cls, x):
        """Wraps a call to `np.random.permutation` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.permutation.html.
        """
        cls._ensure_reproducibility()
        return np.random.permutation(x=x)
    
    @classmethod
    def shuffle_reindex_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffles a dataframe and resets its index.

        Args:
            df (pd.DataFrame): the pandas dataframe to shuffle.

        Returns:
            pd.DataFrame: the shuffled pandas dataframe.
        """
        # pandas uses numpy's random seed internally: https://stackoverflow.com/a/52375474
        cls._ensure_reproducibility() 
        return df.sample(frac=1, replace=False).reset_index(drop=True)
