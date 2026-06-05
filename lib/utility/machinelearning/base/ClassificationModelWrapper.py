from sklearn.pipeline import Pipeline

from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper
from lib.utility.machinelearning.evaluation.Metrics import Metrics


class ClassificationModelWrapper(BaseModelWrapper):
    """
    Base wrapper for all classification models.
    """
    task = "classification"

    def build_pipeline(self, preprocessor):
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

    def evaluate(self, y_true, y_pred):
        return Metrics.classification(y_true, y_pred)
