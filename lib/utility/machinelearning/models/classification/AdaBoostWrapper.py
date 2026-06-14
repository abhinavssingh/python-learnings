from sklearn.ensemble import AdaBoostClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class AdaBoostWrapper(ClassificationModelWrapper):
    """
    Wrapper for AdaBoost classification model.
    """

    def __init__(self):
        super().__init__(AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42))

        self.family = "boosting"
        # ✅ Tuning support
        self.param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0]
        }
