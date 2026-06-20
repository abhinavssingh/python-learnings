from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from lib.utility.machinelearning.base.EnsembleModelWrapper import EnsembleModelWrapper


class SequentialEnsembleWrapper(EnsembleModelWrapper):

    def __init__(self, method=None, model=None):
        super().__init__()

        self.method = method
        self.custom_model = model

    def _build_model(self):

        # ✅ ALWAYS check None explicitly (CRITICAL FIX)
        if self.custom_model is not None:
            return self.custom_model

        if self.method == "adaboost":
            return AdaBoostClassifier()

        elif self.method == "gradient_boosting":
            return GradientBoostingClassifier()

        else:
            raise ValueError("Unsupported sequential method")

    def build_pipeline(self, preprocessor):

        steps = []

        if hasattr(preprocessor, "steps"):
            steps.extend(preprocessor.steps)
        else:
            steps.append(("preprocessor", preprocessor))

        if self.imbalance_handler:
            steps.append(self.imbalance_handler.get_pipeline_step())

        self.model = self._build_model()
        steps.append(("model", self.model))

        self.pipeline = Pipeline(steps)
