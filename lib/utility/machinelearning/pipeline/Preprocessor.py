from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ✅ Ensure DataFrame output globally
set_config(transform_output="pandas")


class Preprocessor:
    """
    Builds reusable preprocessing pipeline.
    Supports optional imputer + outlier handler.
    """

    def __init__(self, X, imputer=None, outlier_handler=None):
        self.X = X
        self.imputer = imputer
        self.outlier_handler = outlier_handler

    def build(self):

        num_cols = self.X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = self.X.select_dtypes(include=["object", "category", "string"]).columns

        # ✅ Numeric pipeline
        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        # ✅ Categorical pipeline
        categorical_pipeline = Pipeline([
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        # ✅ Column transformer
        column_transform = ColumnTransformer(
            [
                ("num", numeric_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols)
            ],
            remainder="drop"
        )

        steps = []

        # ✅ Optional preprocessing
        if self.imputer:
            steps.append(("imputer", self.imputer))

        if self.outlier_handler:
            steps.append(("outlier", self.outlier_handler))

        # ✅ Final transformer
        steps.append(("preprocessor", column_transform))

        return Pipeline(steps)
