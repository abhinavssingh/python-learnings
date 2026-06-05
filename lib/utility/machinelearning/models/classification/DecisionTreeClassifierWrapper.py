from sklearn.tree import DecisionTreeClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class DecisionTreeClassifierWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(DecisionTreeClassifier(random_state=42))
