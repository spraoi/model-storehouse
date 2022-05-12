import numpy as np
import pandas as pds
from typing import List
#joblib needs this class
class CategoricalGrouper:

    """we are fitting the categorical grouper on a reference data
    using min_freq, which means all the low-frequency categories are
    pushed into a garbage or OOV category"""

    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.is_fitted = False


    def fit(self, X, categorical: List):
        _non_oov_dict = dict()
        for col in categorical:
            _df = pd.DataFrame(X[col].value_counts()).reset_index(drop=False)
            _df.columns = ["index", "counts"]
            non_oov = list(_df.loc[_df['counts'] >= self.min_freq, "index"])
            _non_oov_dict[col] = non_oov
#             X.loc[~X[col].isin(non_oov), col] = 'OOV'
        self._non_oov_dict = _non_oov_dict
        self.is_fitted = True
        return None


    def transform(self, X, categorical: List):
        if self.is_fitted:
            for col in categorical:
                non_oov = self._non_oov_dict[col]
                X.loc[~X[col].isin(non_oov), col] = 'OOV'

            return X
        else:
            raise ValueError(
                "The categorical grouper needs to be fit before calling .transform()"
            )

