from sklearn.linear_model import LogisticRegression

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class LogisticRegressionWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(LogisticRegression())
