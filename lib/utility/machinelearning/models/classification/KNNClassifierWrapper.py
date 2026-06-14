from sklearn.neighbors import KNeighborsClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class KNNClassifierWrapper(ClassificationModelWrapper):
    """
    Wrapper for K-Nearest Neighbors classification model.
    """

    def __init__(self):
        super().__init__(
            KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean', n_jobs=8))

        self.family = "instance"

        # ✅ Tuning support
        self.param_grid = {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        }
