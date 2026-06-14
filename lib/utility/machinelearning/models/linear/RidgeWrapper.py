from sklearn.linear_model import Ridge

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class RidgeWrapper(RegressionModelWrapper):
    """
    Wrapper for Ridge Regression model.
    """

    def __init__(self):
        super().__init__(Ridge(alpha=1.0, random_state=42))

        self.family = "linear"

        # ✅ Optional tuning support
        self.param_grid = {
            "model__alpha": [0.01, 0.1, 1.0, 10, 100]
        }
