from sklearn.linear_model import Lasso

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class LassoWrapper(RegressionModelWrapper):
    """
    Wrapper for Lasso Regression model.
    """

    def __init__(self):
        super().__init__(Lasso(alpha=1.0, random_state=42))

        self.family = "linear"

        # ✅ Optional (for AutoML / tuning)
        self.param_grid = {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10]
        }
