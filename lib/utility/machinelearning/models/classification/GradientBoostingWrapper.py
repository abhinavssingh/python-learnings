from sklearn.ensemble import GradientBoostingClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class GradientBoostingWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(model=GradientBoostingClassifier())
