from typing import List, Optional, Tuple


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _RandomSeedOperations:
    _random_seed: int | float = None


class ReproducibleOperations(_RandomSeedOperations, metaclass=Singleton):

    @classmethod
    def _ensure_reproducibility(self) -> None:
        import numpy as np
        from synqtab.reproducibility import ReproducibilityError
        
        if self._random_seed:
            np.random.seed(self._random_seed)
            return
        raise ReproducibilityError(
            f"The reproducibility of the requested operation that involves randomness cannot be ensured. \
                Make sure to call the set_random_seed(some_seed) function at least once in your program."
        )

    @classmethod
    def set_random_seed(cls, random_seed: int | float):
        cls._random_seed = random_seed

    @classmethod
    def get_current_random_seed(cls) -> int:
        return cls._random_seed

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
        if not elements:
            return []
        
        import numpy as np
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
        import numpy as np
        
        cls._ensure_reproducibility()
        return np.random.uniform(low=low, high=high, size=size)

    @classmethod
    def normal(cls, loc: float, scale: float, size: int | Tuple[int]):
        """Wraps a call to `np.random.normal` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.normal.html.
        """
        import numpy as np
        
        cls._ensure_reproducibility()
        return np.random.normal(loc=loc, scale=scale, size=size)
    
    @classmethod
    def permutation(cls, x):
        """Wraps a call to `np.random.permutation` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.permutation.html.
        """
        import numpy as np
        
        cls._ensure_reproducibility()
        return np.random.permutation(x=x)
    
    @classmethod
    def shuffle_reindex_dataframe(cls, df):
        """Shuffles a dataframe and resets its index.

        Args:
            df (pd.DataFrame): the pandas dataframe to shuffle.

        Returns:
            pd.DataFrame: the shuffled pandas dataframe.
        """
        # pandas uses numpy's random seed internally: https://stackoverflow.com/a/52375474
        cls._ensure_reproducibility() 
        return df.sample(frac=1, replace=False).reset_index(drop=True)

    @classmethod
    def train_test_split(cls, *arrays, test_size=None, train_size=None, shuffle=True, stratify=None):
        """Performs train/test split leveraging sklearns respective implementation:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
        The splitting is performed with the appropriate random seed for reproducibility. For details
        on the parameters, please see the original implementation.

        Args:
            test_size (float | int, optional): See original implementation. Defaults to None.
            train_size (float | int, optional): See original implementation. Defaults to None.
            shuffle (bool, optional): See original implementation. Defaults to True.
            stratify (array-like, optional): See original implementation. Defaults to None.

        Returns:
            list: The splitting as returned by sklearn's implementation. 2 * len(arrays) arrays.
        """
        from sklearn.model_selection import train_test_split
        
        return train_test_split(
            arrays,
            test_size=test_size,
            train_size=train_size,
            shuffle=True,
            stratify=stratify,
            random_state=cls._random_seed,
        )

    @classmethod
    def get_isolation_forest_model(cls, n_estimators: int = 100, contamination: float | str = "auto"):
        """Returns an Isolation Forest model with the appropriate random seed. Leverages
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

        Args:
            n_estimators (int, optional): See original implementation for details. Defaults to 100.
            contamination (float | str, optional): See original implementation for details. Defaults to "auto".

        Returns:
            sklearn.ensemble.IsolationForest: an IsolationForest model pre-initialized with the appropriate random seed.
        """
        from sklearn.ensemble import IsolationForest
        
        return IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=cls._random_seed
        )
        
    @classmethod
    def get_random_forest_regressor(cls, n_estimators: int=100):
        """Returns an Random Forest regressor with the appropriate random seed. Leverages
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        Args:
            n_estimators (int, optional): See original implementation for details. Defaults to 100.

        Returns:
            sklearn.ensemble.RandomForestRegressor: an RandomForestRegressor model pre-initialized with the appropriate random seed.
        """
        from sklearn.ensemble import RandomForestRegressor
        
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=cls._random_seed,
            n_jobs=-1
        )
    
    @classmethod
    def get_tabpfn_classifier_model(cls):
        from tabpfn_extensions import TabPFNClassifier
        
        return TabPFNClassifier(random_state=cls._random_seed)
    
    @classmethod
    def get_tabpfn_regression_model(cls):
        from tabpfn_extensions import TabPFNRegressor
        
        return TabPFNRegressor(random_state=cls._random_seed)
    
    @classmethod
    def get_tabebm_model(cls):
        from tabpfn_extensions.tabebm.tabebm import TabEBM, seed_everything
        
        cls._ensure_reproducibility()
        seed_everything(cls._random_seed)
        return TabEBM()
    
    @classmethod
    def get_realtabformer_model(cls, model_type='tabular', gradient_accumulation_steps=4, logging_steps=100):
        from realtabformer import REaLTabFormer
        
        return REaLTabFormer(
            model_type=model_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=logging_steps,
            random_state=cls._random_seed
        )
