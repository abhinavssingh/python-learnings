from xgboost import XGBClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class XGBoostWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(
            model=XGBClassifier(eval_metric="logloss"))
