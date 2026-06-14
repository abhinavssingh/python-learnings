from sklearn.ensemble import GradientBoostingClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class GradientBoostingWrapper(ClassificationModelWrapper):
    """
    Wrapper for Gradient Boosting classification model.
    """

    def __init__(self):
        super().__init__(GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42))

        self.family = "boosting"

        # ✅ Tuning support
        self.param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5]
        }
