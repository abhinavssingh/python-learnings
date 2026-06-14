from sklearn.linear_model import LinearRegression

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class LinearRegressionWrapper(RegressionModelWrapper):
    """
    Wrapper for Linear Regression model.
    """

    def __init__(self):
        super().__init__(LinearRegression(n_jobs=None))

        self.family = "linear"

        # ✅ Optional tuning support
        self.param_grid = {
            "model__fit_intercept": [True, False],
            "model__positive": [False, True]
        }
