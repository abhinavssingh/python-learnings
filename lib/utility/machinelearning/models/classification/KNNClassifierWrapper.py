from sklearn.neighbors import KNeighborsClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class KNNClassifierWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(KNeighborsClassifier(n_jobs=8, n_neighbors=5))
