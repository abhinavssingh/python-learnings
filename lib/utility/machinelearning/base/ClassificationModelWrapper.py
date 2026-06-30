from imblearn.pipeline import Pipeline

from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper
from lib.utility.machinelearning.evaluation.Metrics import Metrics


class ClassificationModelWrapper(BaseModelWrapper):

    task = "classification"
    family = "general"

    def _flatten_steps(self, transformer):
        """
        Recursively flatten pipeline steps.
        """
        steps = []

        if hasattr(transformer, "steps"):
            for name, step in transformer.steps:
                # ✅ Recursively flatten
                steps.extend(self._flatten_steps(step))
        else:
            # ✅ Leaf node
            steps.append((type(transformer).__name__.lower(), transformer))

        return steps

    def build_pipeline(self, preprocessor, extra_steps=None):

        steps = []

        # ✅ FULL FLATTEN (robust fix)
        if preprocessor:
            steps.extend(self._flatten_steps(preprocessor))

        # ✅ Optional custom steps
        if extra_steps:
            steps.extend(extra_steps)

        # ✅ SMOTE
        if self.imbalance_handler:
            steps.append(self.imbalance_handler.get_pipeline_step())

        # ✅ Model
        steps.append(("model", self.model))

        self.pipeline = Pipeline(steps)

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
