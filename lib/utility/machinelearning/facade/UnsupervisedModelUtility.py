import copy
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from lib.utility.logger import Logger
from lib.utility.machinelearning.inference.InferenceFactory import InferenceFactory
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.ResultBuilder import ResultBuilder


class UnsupervisedModelUtility:

    def __init__(self, X, imputer=None, outlier_handler=None):
        """
        ✅ Unsupervised utility expects pre-structured feature data
        """

        if X is None:
            raise ValueError("X (input features) must be provided")

        self.X = X

        self.imputer = imputer
        self.outlier_handler = outlier_handler

        self.results = []
        self.registry = ModelRegistry()

        self.preprocessor = None
        self.trained_models = {}

        # ✅ store labels separately (good design ✅)
        self.labels_store = {}

    # ======================================================
    # ✅ PREPARE DATA (CONSISTENCY WITH CLASSIFICATION ✅)
    # ======================================================

    def prepare_data(self):
        """
        ✅ Prepare unsupervised data

        Assumes:
        - X is already prepared (no raw df handling)
        """

        # ✅ ----------------------------------
        # Validate input
        # ✅ ----------------------------------
        if not isinstance(self.X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        if self.X.shape[0] == 0:
            raise ValueError("X is empty")

        # ✅ store feature names (important ✅)
        self.feature_names = list(self.X.columns)

        # ✅ ----------------------------------
        # Build preprocessor (pipeline ONLY ✅)
        # ✅ ----------------------------------
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
            wrapper.build_pipeline(self.preprocessor)

            pipeline = wrapper.get_pipeline()

            if hasattr(pipeline, "fit_predict"):
                output = pipeline.fit_predict(self.X)
            else:
                pipeline.fit(self.X)
                output = pipeline.predict(self.X)

            raw_metrics = {}
            extra = {}
            labels = None

            if wrapper.family == "clustering":

                labels = output

                # ✅ evaluation using SAME processed pipeline
                X_processed = pipeline[:-1].transform(self.X) if hasattr(pipeline, "__getitem__") else self.X

                raw_metrics = wrapper.evaluate(X_processed, labels)

                # ✅ lightweight metadata
                extra = {
                    "n_clusters": len(set(labels)),
                    "noise_points": int((labels == -1).sum()) if -1 in labels else 0
                }

            elif wrapper.family == "dimensionality":

                raw_metrics = {}

                extra = {
                    "output_shape": output.shape,
                    "n_components": output.shape[1]
                }

            metrics = self._normalize_metrics(raw_metrics)

            exp_id = f"{model_name} | unsupervised"

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                experiment=exp_id,
                result_type="unsupervised",
                mode="fit_predict",
                extra=extra,
                **metrics
            )

            # ✅ ✅ STORE LABELS CORRECTLY (FIX ✅)
            if labels is not None:
                self.labels_store[exp_id] = labels

        except Exception as e:

            exp_id = f"{model_name} | unsupervised"

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                experiment=exp_id,
                result_type="failed",
                extra={"error": str(e)}
            )

        # ✅ STORE RESULT
        self.results.append(result)

        # ✅ STORE MODEL
        exp_id = result["experiment"]
        self.trained_models[exp_id] = {
            "wrapper": wrapper,
            "result": result
        }

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

    # ======================================================
    # ✅ BEST MODEL SELECTION (UNSUPERVISED)
    # ======================================================

    def get_best_model(self, metric=None):

        if not self.results:
            return None

        df = pd.DataFrame(self.results)

        # ✅ Only valid models
        df = df[df["type"] != "failed"]

        if df.empty:
            return None

        # ==================================================
        # ✅ CLUSTERING PRIORITY
        # ==================================================
        if metric is None:

            # ✅ Prefer silhouette
            if "silhouette_score" in df.columns:
                metric = "silhouette_score"
                ascending = False

            # ✅ fallback: DB index
            elif "davies_bouldin_score" in df.columns:
                metric = "davies_bouldin_score"
                ascending = True

            # ✅ fallback: CH index
            elif "calinski_harabasz_score" in df.columns:
                metric = "calinski_harabasz_score"
                ascending = False

            else:
                return df.iloc[0].to_dict()

        else:
            # ✅ user provided metric
            if metric.lower() in ["davies_bouldin", "davies_bouldin_score"]:
                metric = "davies_bouldin_score"
                ascending = True
            else:
                ascending = False

        if metric not in df.columns:
            raise ValueError(f"{metric} not found")

        # ✅ rank and select best
        df_sorted = df.sort_values(metric, ascending=ascending)

        best = df_sorted.iloc[0].to_dict()

        # ✅ remove heavy fields
        best.pop("artifacts", None)

        return best

    # ======================================================
    # ✅ Plot Data Extraction (UNSUPERVISED)
    # ======================================================

    def get_plot_data(self):

        plot_rows = []

        for r in self.results:

            row = {
                "model": r.get("model"),
                "family": r.get("family"),
                "experiment": r.get("experiment"),
            }

            # ==================================================
            # ✅ CLUSTERING METRICS
            # ==================================================
            if r.get("family") == "clustering":

                row.update({
                    "silhouette": r.get("silhouette_score"),
                    "davies_bouldin": r.get("davies_bouldin_score"),
                    "calinski_harabasz": r.get("calinski_harabasz_score"),
                    "n_clusters": r.get("n_clusters"),
                    "noise_points": r.get("noise_points"),
                })

                # ✅ attach labels (IMPORTANT)
                labels = self.labels_store.get(r.get("experiment"))
                if labels is not None:
                    row["labels"] = labels

            # ==================================================
            # ✅ DIMENSIONALITY METRICS
            # ==================================================
            elif r.get("family") == "dimensionality":

                row.update({
                    "n_components": r.get("n_components"),
                    "output_shape": r.get("output_shape"),
                })

                # ✅ embeddings for plotting
                embeddings = self.labels_store.get(r.get("experiment"))
                if embeddings is not None:
                    row["embeddings"] = embeddings

            plot_rows.append(row)

        return plot_rows

    # ---------------------------------------------------
    # MODEL PERSISTENCE
    # ---------------------------------------------------

    def save_model(self, exp_id, path):

        if exp_id not in self.trained_models:
            raise ValueError(f"{exp_id} not found")

        model_obj = self.trained_models[exp_id]

        if "pipeline" in model_obj:
            pipeline = model_obj["pipeline"]
            result = model_obj["result"]
        else:
            wrapper = model_obj["wrapper"]
            pipeline = wrapper.get_pipeline()
            result = model_obj["result"]

        # ✅ ensure directory exists ONCE
        os.makedirs(path, exist_ok=True)

        # ✅ save pipeline
        joblib.dump(pipeline, os.path.join(path, "pipeline.pkl"))

        # ✅ metadata

        metadata = {
            "model": result.get("model"),
            "task": "unsupervised",
            "family": result.get("family"),
            "experiment": exp_id,
            "feature_names": list(self.X.columns),
            "inference_version": "v1",
            "pipeline_type": "sklearn_pipeline",
            "validated": False,
            "extra": result.get("extra", {}),
            "n_features": len(self.X.columns)
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # ✅ SAVE LABELS (FIXED)
        labels = self.labels_store.get(exp_id)

        if result.get("family") == "clustering" and labels is not None:
            np.save(os.path.join(path, "labels.npy"), np.asarray(labels))

        Logger.info(f"✅ Model saved: {exp_id}")

    def validate_inference_pipeline(self, exp_id, model_path):
        if exp_id not in self.trained_models:
            raise ValueError(f"{exp_id} not found")

        if exp_id not in self.labels_store:
            raise ValueError(f"No stored labels found for {exp_id}")

        # ✅ TRAINING LABELS (stored, DO NOT recompute)
        train_labels = self.labels_store[exp_id]

        # ✅ INFERENCE LABELS
        inf_model = InferenceFactory.load(model_path)
        inf_labels = inf_model.predict(self.X)

        # ✅ CLUSTER-SAFE COMPARISON
        score = adjusted_rand_score(train_labels, inf_labels)

        return {
            "status": "PASS" if score > 0.99 else "FAIL",
            "adjusted_rand_score": float(score),
            "note": "Permutation-invariant validation (correct for clustering)"
        }
