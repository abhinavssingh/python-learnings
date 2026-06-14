from lightgbm import LGBMClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class LightGBMWrapper(ClassificationModelWrapper):
    """
    Wrapper for LightGBM classification model.
    """

    def __init__(self):
        super().__init__(
            LGBMClassifier(n_estimators=50, learning_rate=0.1, max_depth=-1, random_state=42, n_jobs=8, verbose=-1))

        self.family = "boosting"

        # ✅ Tuning support
        self.param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [-1, 5, 10],
            "model__num_leaves": [31, 50]
        }
