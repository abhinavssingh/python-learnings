from xgboost import XGBClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class XGBoostWrapper(ClassificationModelWrapper):
    """
    Wrapper for XGBoost classification model.
    """

    def __init__(self):
        super().__init__(
            XGBClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=42, n_jobs=8, use_label_encoder=False))

        self.family = "boosting"   # ✅ CRITICAL

        # ✅ Tuning support
        self.param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 6, 10],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
