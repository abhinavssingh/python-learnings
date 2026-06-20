from imblearn.pipeline import Pipeline

from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper
from lib.utility.machinelearning.evaluation.Metrics import Metrics


class UnsupervisedModelWrapper(BaseModelWrapper):

    task = "unsupervised"
    family = "unsupervised"

    def build_pipeline(self, preprocessor):

        steps = []

        # ✅ Flatten preprocessor (align with supervised design)
        if hasattr(preprocessor, "steps"):
            steps.extend(preprocessor.steps)
        else:
            steps.append(("preprocessor", preprocessor))

        steps.append(("model", self.model))

        self.pipeline = Pipeline(steps)

    # ======================================================
    # ✅ FIT + PREDICT (CLUSTERING)
    # ======================================================
    def predict(self, X):

        # ✅ Clustering models use fit_predict
        if hasattr(self.pipeline, "fit_predict"):
            output = self.pipeline.fit_predict(X)

        else:
            # ✅ fallback (rare case)
            self.pipeline.fit(X)

            if hasattr(self.pipeline, "predict"):
                output = self.pipeline.predict(X)
            elif hasattr(self.pipeline, "transform"):
                output = self.pipeline.transform(X)
            else:
                raise AttributeError("Model does not support predict or transform")

        # ✅ store last output for evaluation reuse
        self._last_output = output

        return output

    # ======================================================
    # ✅ EVALUATION (IMPORTANT FIX ✅)
    # ======================================================
    def evaluate(self, X, labels):

        # ✅ get processed data from pipeline (WITHOUT refit)
        try:
            preprocessor = self.pipeline[:-1]  # all steps except model
            X_processed = preprocessor.transform(X)
        except Exception:
            # fallback if slicing not supported
            X_processed = X

        # ✅ clustering evaluation
        if self.family == "clustering":
            return Metrics.unsupervised(X_processed, labels)

        # ✅ dimensionality / embeddings (no metrics yet)
        return {}
