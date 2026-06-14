from sklearn.pipeline import Pipeline

from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper
from lib.utility.machinelearning.evaluation.Metrics import Metrics


class ClassificationModelWrapper(BaseModelWrapper):

    task = "classification"
    family = "general"

    def build_pipeline(self, preprocessor):
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

    def predict_proba(self, X):
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        return None

    def evaluate(self, y_true, y_pred, y_proba=None):

        return Metrics.classification(
            y_true,
            y_pred,
            y_proba=y_proba,
            include_curves=True,
            include_confusion_matrix=True
        )
