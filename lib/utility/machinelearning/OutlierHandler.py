import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles outliers using:
    - IQR method (default)
    - Z-score method

    Includes logging:
    - Outlier counts before & after
    - Configuration tracking
    """

    def __init__(self, method="iqr", factor=1.5, z_thresh=3):
        self.method = method
        self.factor = factor
        self.z_thresh = z_thresh
        self.results = {}  # ✅ logging container

    def get_params(self, deep=True):
        return {
            "method": self.method,
            "factor": self.factor,
            "threshold": self.z_thresh
        }

    # ---------------------------------------------------
    # FIT
    # ---------------------------------------------------
    def fit(self, X, y=None):

        X = X.copy()

        self.num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)

        # ✅ Log BEFORE handling
        self.results['outliers_before'] = {}

        if self.method == "iqr":
            self.bounds = {}

            for col in self.num_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - self.factor * IQR
                upper = Q3 + self.factor * IQR

                self.bounds[col] = (lower, upper)

                # ✅ count outliers
                outliers = ((X[col] < lower) | (X[col] > upper)).sum()
                self.results['outliers_before'][col] = int(outliers)

        elif self.method == "zscore":
            self.stats = {}

            for col in self.num_cols:
                mean = X[col].mean()
                std = X[col].std()

                self.stats[col] = (mean, std)

                # ✅ avoid division by zero
                if std == 0:
                    self.results['outliers_before'][col] = 0
                    continue

                z_scores = (X[col] - mean) / std
                outliers = (np.abs(z_scores) > self.z_thresh).sum()

                self.results['outliers_before'][col] = int(outliers)

        return self

    # ---------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------
    def transform(self, X):

        X = X.copy()

        # ✅ ensure only valid columns
        num_cols = [col for col in self.num_cols if col in X.columns]

        self.results['outliers_after'] = {}

        if self.method == "iqr":
            for col in num_cols:
                lower, upper = self.bounds[col]

                # ✅ count before capping
                before = ((X[col] < lower) | (X[col] > upper)).sum()

                # ✅ cap values
                X[col] = np.where(X[col] < lower, lower, X[col])
                X[col] = np.where(X[col] > upper, upper, X[col])

                # ✅ count after capping
                after = ((X[col] < lower) | (X[col] > upper)).sum()

                self.results['outliers_after'][col] = {
                    "before": int(before),
                    "after": int(after)
                }

        elif self.method == "zscore":
            for col in num_cols:

                mean, std = self.stats[col]

                if std == 0:
                    self.results['outliers_after'][col] = {
                        "before": 0,
                        "after": 0
                    }
                    continue

                z_scores = (X[col] - mean) / std

                before = (np.abs(z_scores) > self.z_thresh).sum()

                # ✅ replace outliers with mean
                X.loc[np.abs(z_scores) > self.z_thresh, col] = mean

                # recompute
                z_scores_new = (X[col] - mean) / std
                after = (np.abs(z_scores_new) > self.z_thresh).sum()

                self.results['outliers_after'][col] = {
                    "before": int(before),
                    "after": int(after)
                }

        # ✅ config log
        self.results['config'] = {
            "method": self.method,
            "factor": self.factor,
            "z_thresh": self.z_thresh
        }

        return X
