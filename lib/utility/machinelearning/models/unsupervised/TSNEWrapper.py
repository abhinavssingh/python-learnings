from sklearn.manifold import TSNE

from lib.utility.machinelearning.base.UnsupervisedModelWrapper import UnsupervisedModelWrapper


class TSNEWrapper(UnsupervisedModelWrapper):

    def __init__(self, n_components=2):
        super().__init__(model=TSNE(n_components=n_components))
        self.family = "dimensionality"

    def predict(self, X):
        return self.pipeline.fit_transform(X)
