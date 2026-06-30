import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


from lib.utility.machinelearning._logging import ExceptionLoggingMixin


class FeatureNameCleaner(ExceptionLoggingMixin, BaseEstimator, TransformerMixin):
    """
    Cleans feature names to make them XGBoost compatible
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X.columns = [
                re.sub(r"[\\[\\]<>]", "", col) for col in X.columns
            ]
        return X

