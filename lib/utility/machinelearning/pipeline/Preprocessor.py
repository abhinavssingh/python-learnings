from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lib.utility.machinelearning._logging import ExceptionLoggingMixin

# ✅ Ensure DataFrame output globally
set_config(transform_output="pandas")


class Preprocessor(ExceptionLoggingMixin):
    """
    Builds reusable preprocessing pipeline.
    Supports optional imputer + outlier handler.
    """

    def __init__(self, X, imputer=None, outlier_handler=None, mode="supervised"):
        self.X = X
        self.imputer = imputer
        self.outlier_handler = outlier_handler
        self.mode = mode

    def build(self):

        # Include all numeric types after dataset optimization (e.g., uint8, float16).
        num_cols = self.X.select_dtypes(include=["number", "bool"]).columns
        cat_cols = self.X.select_dtypes(include=["object", "category", "string"]).columns

        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        # ✅ SAME for all tasks
        categorical_pipeline = Pipeline([
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        column_transform = ColumnTransformer(
            [
                ("num", numeric_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols)
            ],
            remainder="drop"
        )

        steps = []

        if self.imputer:
            steps.append(("imputer", self.imputer))

        if self.outlier_handler:
            steps.append(("outlier", self.outlier_handler))

        steps.append(("preprocessor", column_transform))

        return Pipeline(steps)
