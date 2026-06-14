from sklearn.ensemble import AdaBoostClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class AdaBoostWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(model=AdaBoostClassifier())
