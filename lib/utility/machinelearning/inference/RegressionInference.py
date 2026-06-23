from .BaseInferencePipeline import BaseInferencePipeline


class RegressionInference(BaseInferencePipeline):

    def predict(self, X):
        X = self._prepare_input(X)
        return self.pipeline.predict(X)
