from sklearn.svm import SVC

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class SVCWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(SVC(probability=True, kernel='linear', random_state=42))
