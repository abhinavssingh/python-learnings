import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, num_strategy="mean", cat_strategy="mode", groupby_cols=None):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.groupby_cols = groupby_cols

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        self.results_ = {}

        self.num_cols = list(X.select_dtypes(include=["int64", "float64"]).columns)
        self.cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)

        # ✅ safe groupby cols
        if self.groupby_cols:
            valid_group_cols = [col for col in self.groupby_cols if col in X.columns]
            self._groupby_cols = valid_group_cols if valid_group_cols else None
        else:
            self._groupby_cols = None

        # ✅ global numeric
        self.global_num_values_ = {}
        for col in self.num_cols:
            if self.num_strategy == "mean":
                self.global_num_values_[col] = X[col].mean()
            elif self.num_strategy == "median":
                self.global_num_values_[col] = X[col].median()

        # ✅ global categorical
        self.global_cat_values_ = {}
        for col in self.cat_cols:
            if self.cat_strategy == "mode":
                mode_val = X[col].mode()
                self.global_cat_values_[col] = mode_val.iloc[0] if not mode_val.empty else None
            elif self.cat_strategy == "constant":
                self.global_cat_values_[col] = "missing"

        return self

    def transform(self, X):

        check_is_fitted(self, ["num_cols", "cat_cols"])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        num_cols = [c for c in self.num_cols if c in X.columns]
        cat_cols = [c for c in self.cat_cols if c in X.columns]

        # ✅ group imputation
        if self._groupby_cols:

            for col in num_cols:
                if self.num_strategy == "mean":
                    X[col] = X.groupby(self._groupby_cols)[col].transform(lambda x: x.fillna(x.mean()))
                elif self.num_strategy == "median":
                    X[col] = X.groupby(self._groupby_cols)[col].transform(lambda x: x.fillna(x.median()))

            for col in cat_cols:
                X[col] = X.groupby(self._groupby_cols)[col].transform(
                    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else None)
                )

        # ✅ global fallback
        for col in num_cols:
            X[col] = X[col].fillna(self.global_num_values_.get(col, None))

        for col in cat_cols:
            X[col] = X[col].fillna(self.global_cat_values_.get(col, None))

        return X
