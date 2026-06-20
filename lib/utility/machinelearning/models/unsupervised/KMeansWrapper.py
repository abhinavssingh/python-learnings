from sklearn.cluster import KMeans

from lib.utility.machinelearning.base.UnsupervisedModelWrapper import UnsupervisedModelWrapper


class KMeansWrapper(UnsupervisedModelWrapper):

    def __init__(self, n_clusters=3):
        super().__init__(model=KMeans(n_clusters=n_clusters))
        self.family = "clustering"
