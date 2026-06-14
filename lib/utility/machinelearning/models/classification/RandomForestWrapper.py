from sklearn.ensemble import RandomForestClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class RandomForestClassifierWrapper(ClassificationModelWrapper):
    """
    Wrapper for Random Forest classification model.
    """

    def __init__(self):
        super().__init__(
            RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=8, random_state=42))

        self.family = "tree"

        # ✅ Optional tuning support
        self.param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 10],
            "model__min_samples_split": [2, 5, 10]
        }
