import os

import numpy as np

from .BaseInferencePipeline import BaseInferencePipeline


class UnsupervisedInference(BaseInferencePipeline):

    def __init__(self, pipeline, metadata, model_path):
        super().__init__(pipeline, metadata)

        self.labels = None
        labels_path = os.path.join(model_path, "labels.npy")

        if os.path.exists(labels_path):
            self.labels = np.load(labels_path)

    def predict(self, X):
        X = self._prepare_input(X)

        if hasattr(self.pipeline, "predict"):
            return self.pipeline.predict(X)

        # fallback
        if hasattr(self.pipeline, "fit_predict"):
            return self.pipeline.fit_predict(X)

        raise ValueError("Model does not support prediction")

    def get_labels(self):
        return self.labels
