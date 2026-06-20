import copy

import pandas as pd

from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.ResultBuilder import ResultBuilder


class UnsupervisedModelUtility:

    def __init__(self, df, imputer=None, outlier_handler=None):

        self.df = df
        self.imputer = imputer
        self.outlier_handler = outlier_handler

        self.results = []
        self.registry = ModelRegistry()

        # ✅ internal state
        self.X = None
        self.labels_store = {}   # ✅ store labels OUTSIDE results

        self.preprocessor = None

    # ======================================================
    # ✅ PREPARE DATA (CONSISTENCY WITH CLASSIFICATION ✅)
    # ======================================================
    def prepare_data(self):

        df = self.df.copy()

        # ✅ no target, no split
        self.X = df

        # ✅ build pipeline (same pattern ✅)
        self.preprocessor = Preprocessor(
            self.X,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler,
            mode="unsupervised"
        ).build()

        return self

    # ======================================================
    # ✅ RUN SINGLE MODEL (LIKE run_experiment ✅)
    # ======================================================

    def run_experiment(self, model_name):

        wrapper = copy.deepcopy(self.registry.get_model(model_name))

        try:
            # ==================================================
            # ✅ BUILD PIPELINE
            # ==================================================
            wrapper.build_pipeline(self.preprocessor)

            # ==================================================
            # ✅ FIT + PREDICT
            # ==================================================
            output = wrapper.predict(self.X)

            # ✅ processed data for evaluation
            X_processed = self.preprocessor.transform(self.X)

            raw_metrics = {}
            extra = {}
            labels = None

            # ==================================================
            # ✅ CLUSTERING FLOW
            # ==================================================
            if wrapper.family == "clustering":

                labels = output

                # ✅ evaluation (same style as classification)
                raw_metrics = wrapper.evaluate(X_processed, labels)

                # ✅ meta info (keep lightweight ✅)
                extra = {
                    "n_clusters": len(set(labels)),
                    "noise_points": int((labels == -1).sum()) if -1 in labels else 0
                }

                # ✅ store labels separately (NOT in results ✅)
                self.labels_store[model_name] = labels

            # ==================================================
            # ✅ DIMENSIONALITY FLOW
            # ==================================================
            elif wrapper.family == "dimensionality":

                raw_metrics = {}

                extra = {
                    "output_shape": output.shape,
                    "n_components": output.shape[1]
                }

            # ==================================================
            # ✅ NORMALIZE OUTPUT (ALIGN WITH CLASSIFICATION ✅)
            # ==================================================
            metrics = self._normalize_metrics(raw_metrics)

            # ==================================================
            # ✅ BUILD RESULT (CLEAN ✅)
            # ==================================================
            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                result_type="unsupervised",
                mode="fit_predict",
                extra=extra,
                **metrics
            )

        except Exception as e:

            # ✅ failure-safe execution (aligned ✅)
            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                result_type="failed",
                extra={"error": str(e)}
            )

        self.results.append(result)
        return result

    # ======================================================
    # ✅ RUN MULTIPLE MODELS
    # ======================================================
    def run_all_models(self, model_names):

        results = []

        for model_name in model_names:
            res = self.run_experiment(model_name)
            results.append(res)

        return results

    # ======================================================
    # ✅ GET RESULTS
    # ======================================================
    def get_results_df(self):
        return pd.DataFrame(self.results)

    # ======================================================
    # ✅ GET LABELS (FOR VISUALIZATION ✅)
    # ======================================================
    def get_labels(self, model_name):
        return self.labels_store.get(model_name)

    # ======================================================
    # ✅ INTERNAL: NORMALIZE METRICS (KEEP RESULTS CLEAN ✅)
    # ======================================================

    def _normalize_metrics(self, metrics):

        clean_metrics = {}

        for k, v in metrics.items():
            # ✅ keep only scalar values
            if isinstance(v, (int, float)) or v is None:
                clean_metrics[k] = v

        return clean_metrics
