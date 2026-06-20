from sklearn.cluster import DBSCAN

from lib.utility.machinelearning.base.UnsupervisedModelWrapper import UnsupervisedModelWrapper


class DBSCANWrapper(UnsupervisedModelWrapper):

    def __init__(self, eps=0.5):
        super().__init__(model=DBSCAN(eps=eps))
        self.family = "clustering"
