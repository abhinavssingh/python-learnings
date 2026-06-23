import pandas as pd


class BaseInferencePipeline:

    def __init__(self, pipeline, metadata):
        self.pipeline = pipeline
        self.meta = metadata
        self.features = metadata.get("feature_names")

    def _prepare_input(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.features:
            X = X[self.features]

        return X
