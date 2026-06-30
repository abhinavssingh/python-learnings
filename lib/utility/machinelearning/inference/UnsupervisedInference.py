import os

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .BaseInferencePipeline import BaseInferencePipeline


class UnsupervisedInference(BaseInferencePipeline):

    def __init__(self, pipeline, metadata, model_path):
        super().__init__(pipeline, metadata)

        self.labels = None
        labels_path = os.path.join(model_path, "labels.npy")

        if os.path.exists(labels_path):
            self.labels = np.load(labels_path)

    # ------------------------------------------------------------
    # ✅ PREDICT (SAFE + CONSISTENT)
    # ------------------------------------------------------------
    def predict(self, X):

        # ✅ Prepare input (handled in base class)
        X = self._prepare_input(X)

        # ✅ Ensure pipeline is fitted
        try:
            check_is_fitted(self.pipeline)
        except Exception:
            raise ValueError("Loaded pipeline is not fitted")

        # ✅ Use predict (DO NOT refit)
        if hasattr(self.pipeline, "predict"):
            return self.pipeline.predict(X)

        raise ValueError("Pipeline does not support predict()")

    # ------------------------------------------------------------
    # ✅ STORED TRAIN LABELS ACCESS
    # ------------------------------------------------------------
    def get_labels(self):
        return self.labels
