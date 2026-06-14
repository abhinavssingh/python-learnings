from lightgbm import LGBMClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class LightGBMWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(
            model=LGBMClassifier())
