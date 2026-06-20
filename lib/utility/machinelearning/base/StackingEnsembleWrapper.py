from imblearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier

from lib.utility.machinelearning.base.EnsembleModelWrapper import EnsembleModelWrapper


class StackingEnsembleWrapper(EnsembleModelWrapper):

    def __init__(self, models, final_estimator):
        super().__init__()

        self.models = models
        self.final_estimator = final_estimator

    def _build_model(self):

        estimators = [(name, m) for name, m in self.models]

        return StackingClassifier(
            estimators=estimators,
            final_estimator=self.final_estimator,
            passthrough=False  # ✅ can make configurable later
        )

    def build_pipeline(self, preprocessor):

        steps = []

        # ✅ Flatten preprocessor
        if hasattr(preprocessor, "steps"):
            steps.extend(preprocessor.steps)
        else:
            steps.append(("preprocessor", preprocessor))

        # ✅ SMOTE
        if self.imbalance_handler:
            steps.append(self.imbalance_handler.get_pipeline_step())

        # ✅ Stacking model
        self.model = self._build_model()
        steps.append(("model", self.model))

        self.pipeline = Pipeline(steps)
