import copy
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from lib.utility.logger import Logger
from lib.utility.machinelearning.inference.InferenceFactory import InferenceFactory
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.ResultBuilder import ResultBuilder
from lib.utility.machinelearning.tuning.UnsupervisedHyperparameterTuner import UnsupervisedHyperparameterTuner


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

        self.feature_names = list(self.X.columns)

        self.preprocessor = Preprocessor(
            self.X,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler,
            mode="unsupervised"
        ).build()

        # ✅ ✅ ADD THIS LINE (CRITICAL FIX)
        self.preprocessor.fit(self.X)

        return self

    # ======================================================
    # ✅ RUN SINGLE MODEL (LIKE run_experiment ✅)
    # ======================================================

    def run_experiment(self, model_name):

        wrapper = copy.deepcopy(self.registry.get_model(model_name))

        try:
            # ======================================================
            # ✅ STEP 2: DIMENSIONALITY REDUCTION (CRITICAL ✅)
            # ======================================================
            reducer = self._build_reducer()

            # ======================================================
            # ✅ BUILD FULL TRAINING / INFERENCE PIPELINE
            # ======================================================
            wrapper.build_pipeline(
                self.preprocessor,
                extra_steps=[("reducer", reducer)]
            )
            pipeline = wrapper.get_pipeline()

            # ======================================================
            # ✅ STEP 3: CLUSTERING ON REDUCED SPACE ✅
            # ======================================================
            if hasattr(pipeline, "fit_predict"):
                labels = pipeline.fit_predict(self.X)
            else:
                pipeline.fit(self.X)
                labels = pipeline.predict(self.X)

            # ======================================================
            # ✅ STEP 4: METRICS ON REDUCED SPACE ✅
            # ======================================================
            raw_metrics = wrapper.evaluate(self.X, labels)

            extra = {
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "noise_points": int(np.sum(labels == -1)) if -1 in labels else 0
            }

            metrics = self._normalize_metrics(raw_metrics)

            exp_id = f"{model_name} | unsupervised"

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                experiment=exp_id,
                task="unsupervised",
                mode="fit_predict",
                extra=extra,
                **metrics
            )

            # ======================================================
            # ✅ STORE LABELS (ALIGNED WITH REDUCED DATA ✅)
            # ======================================================
            self.labels_store[exp_id] = labels

            # ✅ ALSO STORE PCA MODEL (IMPORTANT FOR PLOTS ✅)
            if not hasattr(self, "reducers"):
                self.reducers = {}

            self.reducers[exp_id] = reducer

        except Exception as e:

            exp_id = f"{model_name} | unsupervised"

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                experiment=exp_id,
                task="unsupervised",
                result_type="failed",
                extra={"error": str(e)}
            )

        # ======================================================
        # ✅ STORE RESULT + MODEL
        # ======================================================
        self.results.append(result)

        self.trained_models[result["experiment"]] = {
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
    # ✅ HYPERPARAMETER TUNING (UNSUPERVISED)
    # ======================================================

    def tune_model(
        self,
        model_name,
        param_config=None,
        search_type="grid",
        n_iter=20,
        random_state=42,
        **kwargs,
    ):
        if self.preprocessor is None:
            raise ValueError("Call prepare_data() before tune_model()")

        wrapper = copy.deepcopy(self.registry.get_model(model_name))

        param_config = self._resolve_param_config(param_config, kwargs)

        tuner = UnsupervisedHyperparameterTuner(self.X)

        raw_results = tuner.tune(
            wrapper=wrapper,
            model_name=model_name,
            preprocessor=self.preprocessor,
            search_type=search_type,
            param_config=param_config,
            n_iter=n_iter,
            random_state=random_state,
        )

        if not hasattr(self, "reducers"):
            self.reducers = {}

        final_results = []

        for payload in raw_results:
            result = payload["result"]
            exp_id = result["experiment"]

            labels = payload.get("labels")
            reducer = payload.get("reducer")
            pipeline = payload.get("pipeline")

            if labels is not None:
                self.labels_store[exp_id] = labels
            if reducer is not None:
                self.reducers[exp_id] = reducer
            if pipeline is not None:
                self.trained_models[exp_id] = {
                    "pipeline": pipeline,
                    "result": result,
                }

            self.results.append(result)
            final_results.append(result)

        return final_results

    # ======================================================
    # ✅ TUNE ALL MODELS (UNSUPERVISED)
    # ======================================================

    def tune_all_models(
        self,
        param_configs,
        search_type="grid",
        n_iter=20,
        random_state=42,
        **kwargs,
    ):
        """
        Tune multiple unsupervised models.

        param_configs example:
        {
            "KMeans": {"model__n_clusters": [2, 3, 4]},
            "DBSCAN": {"model__eps": [0.3, 0.5], "model__min_samples": [5, 10]}
        }
        """

        all_results = []

        for model_name in param_configs:
            Logger.info(f"🔧 Tuning {model_name}...")

            res = self.tune_model(
                model_name=model_name,
                param_config=param_configs[model_name],
                search_type=search_type,
                n_iter=n_iter,
                random_state=random_state,
                **kwargs,
            )

            all_results.extend(res)

        return pd.DataFrame(all_results)

    # ======================================================
    # ✅ GET RESULTS
    # ======================================================
    def get_results_df(self):
        return pd.DataFrame(self.results)

    # ======================================================
    # ✅ GET LABELS (FOR VISUALIZATION ✅)
    # ======================================================

    def get_labels(self, model_name):
        exp_id = f"{model_name} | unsupervised"
        return self.labels_store.get(exp_id)

    def _resolve_param_config(self, param_config, kwargs):
        """
        Normalize tuning params into sklearn pipeline format.
        Example:
          n_clusters=[2, 3] -> model__n_clusters=[2, 3]
        """
        if param_config is not None:
            return param_config

        if not kwargs:
            return None

        normalized = {}

        for key, value in kwargs.items():
            param_key = key if key.startswith("model__") else f"model__{key}"
            normalized[param_key] = value if isinstance(value, list) else [value]

        return normalized

    def _build_reducer(self):
        """
        Keep PCA robust for low-dimensional inputs.
        """
        n_features = self.X.shape[1]
        n_components = max(1, min(10, n_features))
        return PCA(n_components=n_components, random_state=42)

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

        df = pd.DataFrame(self.results)

        # ✅ Safe filtering (remove failed runs if column exists)
        if "result_type" in df.columns:
            df = df[df["result_type"] != "failed"]

        if df.empty:
            return None

        # ==================================================
        # ✅ METRIC SELECTION
        # ==================================================
        if metric is None:

            if "silhouette_score" in df.columns:
                metric = "silhouette_score"
                ascending = False

            elif "davies_bouldin_score" in df.columns:
                metric = "davies_bouldin_score"
                ascending = True

            elif "calinski_harabasz_score" in df.columns:
                metric = "calinski_harabasz_score"
                ascending = False

            else:
                # fallback: return first valid model
                return df.iloc[0].to_dict()

        else:
            # ✅ normalize metric input
            if metric.lower() in ["davies_bouldin", "davies_bouldin_score"]:
                metric = "davies_bouldin_score"
                ascending = True
            else:
                ascending = False

        # ==================================================
        # ✅ METRIC VALIDATION
        # ==================================================
        if metric not in df.columns or df[metric].isna().all():
            return None

        # ✅ remove rows with invalid metric values
        df = df[df[metric].notna()]

        if df.empty:
            return None

        # ==================================================
        # ✅ SELECT BEST MODEL
        # ==================================================
        df_sorted = df.sort_values(metric, ascending=ascending)

        best = df_sorted.iloc[0].to_dict()

        # ✅ remove heavy/unnecessary fields
        best.pop("artifacts", None)

        return best

    # ======================================================
    # ✅ Plot Data Extraction (UNSUPERVISED)
    # ======================================================

    def get_plot_data(self):

        plot_data = {}

        # ✅ processed data
        X_processed = self.get_processed_data()
        plot_data["X_processed"] = X_processed

        # ✅ clusters
        clusters = {}

        for exp_id, labels in self.labels_store.items():
            model_name = exp_id.split("|")[0].strip()
            clusters[model_name] = labels

        plot_data["clusters"] = clusters

        # ✅ reducers (important for consistent plotting)
        plot_data["reducers"] = getattr(self, "reducers", {})

        return plot_data

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
            "task": result.get("task"),
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

        # ==========================================================
        # ✅ VALIDATE INPUTS
        # ==========================================================
        if exp_id not in self.trained_models:
            raise ValueError(f"{exp_id} not found")

        if exp_id not in self.labels_store:
            raise ValueError(f"No stored labels found for {exp_id}")

        # ==========================================================
        # ✅ TRAINING LABELS
        # ==========================================================
        train_labels = self.labels_store[exp_id]

        # ==========================================================
        # ✅ LOAD INFERENCE MODEL
        # ==========================================================
        inf_model = InferenceFactory.load(model_path)

        # ==========================================================
        # ✅ INFERENCE PREDICTIONS
        # ==========================================================
        inf_labels = inf_model.predict(self.X)

        # ==========================================================
        # ✅ VALIDATIONS
        # ==========================================================

        # ✅ Length check
        if len(train_labels) != len(inf_labels):
            return {
                "status": "FAIL",
                "reason": "Label length mismatch",
                "train_len": len(train_labels),
                "inference_len": len(inf_labels),
            }

        # ✅ Cluster counts (ignore noise = -1)
        train_clusters = len(set(train_labels) - {-1})
        inf_clusters = len(set(inf_labels) - {-1})

        # ==========================================================
        # ✅ METRIC (Permutation-invariant)
        # ==========================================================
        score = adjusted_rand_score(train_labels, inf_labels)

        # ==========================================================
        # ✅ RESULT
        # ==========================================================
        return {
            "status": "PASS" if score > 0.99 else "FAIL",
            "adjusted_rand_score": float(score),
            "train_clusters": int(train_clusters),
            "inference_clusters": int(inf_clusters),
            "note": "Permutation-invariant validation (correct for clustering)"
        }

    def get_processed_data(self):

        if hasattr(self, "_X_processed"):
            return self._X_processed

        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized")

        self._X_processed = self.preprocessor.transform(self.X)
        return self._X_processed
