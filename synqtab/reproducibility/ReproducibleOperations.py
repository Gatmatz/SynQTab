from typing import List, Optional, Tuple

from matplotlib.pylab import seed

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
    def seed_everything(self) -> None:
        """
        Set random seeds for reproducibility across all libraries.

        Args:
            seed: Random seed value
        """
        
        import os
        import random
        import numpy as np
        import torch

        os.environ["PL_GLOBAL_SEED"] = str(self._random_seed)
        random.seed(self._random_seed)
        np.random.seed(self._random_seed)
        torch.manual_seed(self._random_seed)
        torch.cuda.manual_seed_all(self._random_seed)

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

        if len(elements) <= how_many:
            return elements
        
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
    def derangement(cls, x):
        """Wraps a call to `np.random.permutation` for reproducibility purposes. For more info
        see https://numpy.org/devdocs/reference/random/generated/numpy.random.permutation.html.
        The function returns a **derangement**, i.e., there is no element that remains in its
        original position. This is performed by generating consecutive permutations with the same
        random seed, until the first derangement is found and returned.
        """
        if len(x) == 1:
            return x
        
        import numpy as np
        cls._ensure_reproducibility()
        
        while True:
            permutation = np.random.permutation(x=x)
            
            permutation_is_a_derangement = True
            for original_element, permuted_element in zip(x, permutation):
                if original_element == permuted_element:
                    permutation_is_a_derangement = False
                    break
            
            if permutation_is_a_derangement:
                return permutation

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
    def train_test_split(
        cls,
        df,
        problem_type,
        test_size=None,
        train_size=None,
        shuffle=True,
        stratify=None,
    ):
        """Performs train/test split leveraging sklearns respective implementation:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
        The splitting is performed with the appropriate random seed for reproducibility. For details
        on the parameters, please see the original implementation.

        For regression, it splits the target (shuffle) column into 10 bins for stratified sampling.
        To stratify, every bin must have at least 2 members (so one can go to Train and one to Test)

        Args:
            test_size (float | int, optional): See original implementation. Defaults to None.
            train_size (float | int, optional): See original implementation. Defaults to None.
            shuffle (bool, optional): See original implementation. Defaults to True.
            stratify (array-like, optional): See original implementation. Defaults to None.

        Returns:
            list: The splitting as returned by sklearn's implementation. 2 * len(arrays) arrays.
        """
        from sklearn.model_selection import train_test_split
        from synqtab.enums import ProblemType

        if problem_type == ProblemType.CLASSIFICATION or stratify is None:
            return train_test_split(
                df,
                test_size=test_size,
                train_size=train_size,
                shuffle=shuffle,
                stratify=stratify,
                random_state=cls._random_seed,
            )

        # ELSE IF problem_type == ProblemType.REGRESSION:
        import pandas as pd
        import numpy as np

        N_BINS = 10
        stratify_bins = None
        
        # Safety Loop: Reduce bins if any bin has fewer than 2 members
        # (sklearn requires at least 2 to put one in train and one in test)
        while N_BINS > 1:
            stratify_bins = pd.cut(
                stratify,
                bins=N_BINS,
                labels=False,
                include_lowest=True
            )
            # Check the smallest bin count
            if pd.Series(stratify_bins).value_counts().min() >= 2:
                break
            N_BINS -= 1
               
        # If we hit 1 bin, stratification is impossible/useless, 
        # so we set it to None to avoid errors.
        final_stratify = stratify_bins if N_BINS > 1 else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            train_size=train_size,
            shuffle=shuffle,
            stratify=final_stratify,
            random_state=cls._random_seed,
        )

        return train_df, test_df

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
    def get_tabpfn_unsupervised_model(cls):
        from tabpfn_extensions import TabPFNUnsupervisedModel
        
        classifier = cls.get_tabpfn_classifier_model()
        regressor = cls.get_tabpfn_regression_model()
        return TabPFNUnsupervisedModel(classifier, regressor)

    @classmethod
    def get_tabebm_model(cls):
        from tabebm.TabEBM import TabEBM
        
        cls._ensure_reproducibility()
        cls.seed_everything()
        return TabEBM()
    
    @classmethod
    def get_realtabformer_model(cls, model_type='tabular', gradient_accumulation_steps=4, logging_steps=100):
        import uuid
        from realtabformer import REaLTabFormer
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        realtabformer = REaLTabFormer(
            model_type=model_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=logging_steps,
            random_state=cls._random_seed,
            epochs=500,
            batch_size=64,
        )
        realtabformer.experiment_id = f"run_{uuid.uuid4().hex[:6]}"
        return realtabformer
