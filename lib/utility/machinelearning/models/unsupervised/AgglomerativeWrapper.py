from sklearn.cluster import AgglomerativeClustering

from lib.utility.machinelearning.base.UnsupervisedModelWrapper import UnsupervisedModelWrapper


class AgglomerativeWrapper(UnsupervisedModelWrapper):

    def __init__(self, n_clusters=3):
        super().__init__(model=AgglomerativeClustering(n_clusters=n_clusters))
        self.family = "clustering"
