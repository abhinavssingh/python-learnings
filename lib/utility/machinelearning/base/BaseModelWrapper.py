from lib.utility.machinelearning.base.MLModelBase import MLModelBase


class BaseModelWrapper(MLModelBase):

    def __init__(self, model):
        self.model = model
        self.pipeline = None

    def get_pipeline(self):
        if self.pipeline is None:
            raise ValueError("Pipeline not built.")
        return self.pipeline

    def train(self, X, y):
        self.get_pipeline().fit(X, y)

    def predict(self, X):
        return self.get_pipeline().predict(X)
