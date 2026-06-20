from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, VotingClassifier

from lib.utility.machinelearning.base.EnsembleModelWrapper import EnsembleModelWrapper


class ParallelEnsembleWrapper(EnsembleModelWrapper):

    def __init__(self, models, method="voting", voting="soft"):
        super().__init__()

        self.models = models
        self.method = method
        self.voting = voting

    def _build_model(self):

        estimators = [(name, m) for name, m in self.models]

        if self.method == "voting":
            return VotingClassifier(
                estimators=estimators,
                voting=self.voting
            )

        elif self.method == "bagging":
            # single base estimator
            base_model = self.models[0][1]

            return BaggingClassifier(
                estimator=base_model,
                n_estimators=10
            )

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
