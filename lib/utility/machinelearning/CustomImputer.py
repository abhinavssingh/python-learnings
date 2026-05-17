from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):

    """
    Flexible Imputer:
    - global mean / median / mode
    - group-based imputation
    - logging support
    """

    def __init__(self, num_strategy="mean", cat_strategy="mode", groupby_cols=None):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.groupby_cols = groupby_cols
        self.results = {}  # ✅ logging container

    def get_params(self, deep=True):
        return {
            "num_strategy": self.num_strategy,
            "cat_strategy": self.cat_strategy,
            "groupby_cols": self.groupby_cols
        }

    # ---------------------------------------------------
    # FIT
    # ---------------------------------------------------
    def fit(self, X, y=None):

        X = X.copy()

        # ✅ detect columns
        self.num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)
        self.cat_cols = list(X.select_dtypes(include=['object', 'category', 'string']).columns)
        self.all_cols = self.num_cols + self.cat_cols

        # ✅ VALIDATE GROUPBY COLUMNS
        if self.groupby_cols:
            self.groupby_cols = [col for col in self.groupby_cols if col in X.columns]
            if len(self.groupby_cols) == 0:
                self.groupby_cols = None

        # ✅ LOG BEFORE IMPUTATION
        self.results['imputation_details_before'] = {
            "num_cols_with_nulls": X[self.num_cols].isnull().sum().to_dict(),
            "cat_cols_with_nulls": X[self.cat_cols].isnull().sum().to_dict()
        }

        # ---------- GLOBAL STATS ----------
        self.global_num_values = {}
        self.global_cat_values = {}

        for col in self.num_cols:
            if self.num_strategy == "mean":
                self.global_num_values[col] = X[col].mean()
            elif self.num_strategy == "median":
                self.global_num_values[col] = X[col].median()

        for col in self.cat_cols:
            mode_val = X[col].mode()
            self.global_cat_values[col] = mode_val[0] if not mode_val.empty else None

        return self

    # ---------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------
    def transform(self, X):

        X = X.copy()

        num_cols = [col for col in self.num_cols if col in X.columns]
        cat_cols = [col for col in self.cat_cols if col in X.columns]

        # ---------- GROUP IMPUTATION ----------
        if self.groupby_cols:

            for col in num_cols:
                X[col] = X.groupby(self.groupby_cols)[col].transform(
                    lambda x: x.fillna(x.mean())
                )

            for col in cat_cols:
                X[col] = X.groupby(self.groupby_cols)[col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
                )

        # ---------- GLOBAL FALLBACK ----------
        for col in num_cols:
            if col in self.global_num_values:
                X[col] = X[col].fillna(self.global_num_values[col])

        for col in cat_cols:
            if col in self.global_cat_values:
                X[col] = X[col].fillna(self.global_cat_values[col])

        # ✅ LOG AFTER IMPUTATION
        self.results['imputation_details_after'] = {
            "num_cols_with_nulls": X[num_cols].isnull().sum().to_dict(),
            "cat_cols_with_nulls": X[cat_cols].isnull().sum().to_dict()
        }

        # ✅ SUMMARY LOG
        self.results['config'] = {
            "num_strategy": self.num_strategy,
            "cat_strategy": self.cat_strategy,
            "groupby_cols": self.groupby_cols
        }

        return X
