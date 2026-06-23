from .BaseInferencePipeline import BaseInferencePipeline


class ClassificationInference(BaseInferencePipeline):

    def __init__(self, pipeline, metadata):
        super().__init__(pipeline, metadata)
        self.threshold = metadata.get("threshold", 0.5)

    def predict(self, X):
        X = self._prepare_input(X)
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        X = self._prepare_input(X)

        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)

        return None

    def predict_with_threshold(self, X):

        proba = self.predict_proba(X)

        if proba is None:
            return self.predict(X)

        return (proba[:, 1] >= self.threshold).astype(int)
