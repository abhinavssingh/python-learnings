from sklearn.linear_model import LogisticRegression

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class LogisticRegressionWrapper(ClassificationModelWrapper):
    """
    Wrapper for Logistic Regression classification model.
    """

    def __init__(self):
        super().__init__(LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))

        self.family = "linear"

        # ✅ Optional tuning support
        self.param_grid = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"]
        }
