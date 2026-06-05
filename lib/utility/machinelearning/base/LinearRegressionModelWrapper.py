from sklearn.pipeline import Pipeline

from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper
from lib.utility.machinelearning.evaluation.Metrics import Metrics


class RegressionModelWrapper(BaseModelWrapper):
    """
    Base wrapper for all regression models.
    """
    task = "regression"

    def build_pipeline(self, preprocessor):
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

    def evaluate(self, y_true, y_pred):
        return Metrics.regression(y_true, y_pred)
