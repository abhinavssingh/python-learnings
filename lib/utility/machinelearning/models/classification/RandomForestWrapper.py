from sklearn.ensemble import RandomForestClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class RandomForestClassifierWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(RandomForestClassifier(n_jobs=8, random_state=42))
