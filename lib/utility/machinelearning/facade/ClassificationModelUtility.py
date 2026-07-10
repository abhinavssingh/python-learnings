import copy
import json
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from lib.utility.logger import Logger
from lib.utility.machinelearning.base.ParallelEnsembleWrapper import ParallelEnsembleWrapper
from lib.utility.machinelearning.base.SequentialEnsembleWrapper import SequentialEnsembleWrapper
from lib.utility.machinelearning.base.StackingEnsembleWrapper import StackingEnsembleWrapper
from lib.utility.machinelearning.evaluation.ClassificationModelComparator import ClassificationModelComparator
from lib.utility.machinelearning.inference.InferenceFactory import InferenceFactory
from lib.utility.machinelearning.pipeline.imbalance.SMOTEHandler import SMOTEHandler
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.ClassificationFormatter import ClassificationFormatter as cf
from lib.utility.machinelearning.shared.ResultBuilder import ResultBuilder
from lib.utility.machinelearning.tuning.ClassificationHyperparameterTuner import ClassificationHyperparameterTuner
from lib.utility.machinelearning.validation.CrossValidator import CrossValidator


class ClassificationModelUtility:

    def __init__(self, X_train=None, y_train=None, X_test=None,
                 y_test=None, imputer=None, outlier_handler=None, config=None):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.imputer = imputer
        self.outlier_handler = outlier_handler
        self.config = config

        self.results = []
        self.registry = ModelRegistry()

        self.preprocessor = None
        self.problem_type = None
        self.label_encoder = None
        self.trained_models = {}

    # ----------------------------
    # Data Preparation
    # ----------------------------

    def prepare_data(self):
        """
        ✅ Prepare data for training (pre-split data only)

        Assumes:
        - X_train, y_train already provided
        - X_test, y_test optional (for evaluation)
        """

        try:
            # ✅ ----------------------------------
            # Validate input
            # ✅ ----------------------------------
            if self.X_train is None or self.y_train is None:
                raise ValueError("X_train and y_train must be provided (external split required)")

            # ✅ ----------------------------------
            # Label encoding (train → test consistent)
            # ✅ ----------------------------------
            y = self.y_train

            if isinstance(y, pd.Series) and not pd.api.types.is_numeric_dtype(y):
                self.label_encoder = LabelEncoder()

                self.y_train = self.label_encoder.fit_transform(y)

                if self.y_test is not None:
                    self.y_test = self.label_encoder.transform(self.y_test)

            # ✅ ----------------------------------
            # Detect problem type
            # ✅ ----------------------------------
            self.problem_type = type_of_target(self.y_train)

            Logger.info(f"Detected problem type: {self.problem_type}")

            # ✅ ----------------------------------
            # Multilabel sanity check
            # ✅ ----------------------------------
            if self.problem_type == "multilabel-indicator":
                if not isinstance(self.y_train, (pd.DataFrame)):
                    raise ValueError("Multilabel task requires y_train as DataFrame")

            # ✅ ----------------------------------
            # Build preprocessor (ONLY HERE ✅)
            # ✅ ----------------------------------
            self.preprocessor = Preprocessor(
                self.X_train,
                imputer=self.imputer,
                outlier_handler=self.outlier_handler
            ).build()
        except Exception as e:
            Logger.error(f"ClassificationModelUtility.prepare_data failed: {e}")
            raise

    # ----------------------------
    # Imbalance Handling
    # ----------------------------

    def _get_imbalance_handler(self):

        if not self.config or not self.config.imbalance_strategy:
            return None

        im_type = self.config.imbalance_strategy

        if im_type == "smote":
            return SMOTEHandler(**self.config.imbalance_params)

        return None

    # ----------------------------
    # Run Single Experiment
    # ----------------------------
    def run_experiment(self, model_name):

        wrapper = copy.deepcopy(self.registry.get_model(model_name))

        try:

            if self.problem_type == "multilabel-indicator":
                wrapper.model = OneVsRestClassifier(wrapper.model)

            # --------------------------------------------------
            # Imbalance Handling
            # --------------------------------------------------

            imbalance_handler = self._get_imbalance_handler()

            if imbalance_handler:
                wrapper.set_imbalance_handler(imbalance_handler)

            wrapper.build_pipeline(self.preprocessor)

            # --------------------------------------------------
            # Cross Validation
            # --------------------------------------------------

            cv_results = CrossValidator.run(
                estimator=wrapper.pipeline,
                X=self.X_train,
                y=self.y_train,
                problem_type=self.problem_type,
                strategy=self.config.validation_strategy,
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
                scoring=wrapper.get_scoring_metrics()
            )

            # --------------------------------------------------
            # Final Training
            # --------------------------------------------------

            wrapper.train(self.X_train, self.y_train)

            # --------------------------------------------------
            # Class Distribution
            # --------------------------------------------------

            class_dist_before = None

            if self.problem_type != "continuous":
                class_dist_before = dict(
                    Counter(self.y_train)
                )

            imbalance_summary = None

            if wrapper.imbalance_handler:
                imbalance_summary = (
                    wrapper.imbalance_handler.get_summary()
                )

            # --------------------------------------------------
            # Predictions
            # --------------------------------------------------

            y_pred = wrapper.predict(self.X_test)

            try:
                y_proba = wrapper.predict_proba(self.X_test)
            except Exception:
                y_proba = None

            metrics = wrapper.evaluate(
                self.y_test,
                y_pred,
                y_proba
            )

            artifacts, metrics = self._extract_artifacts(metrics)

            # --------------------------------------------------
            # Cross Validation Artifact
            # --------------------------------------------------

            if cv_results:

                cv_metrics = {}

                for metric, values in cv_results.items():

                    cv_metrics[metric] = values["mean"]
                    cv_metrics[f"{metric}_std"] = values["std"]

                cv_result = ResultBuilder.build(
                    model=model_name,
                    family=getattr(wrapper, "family", "unknown"),
                    result_type="cross_validation",
                    cv_strategy=self.config.validation_strategy,
                    cv_folds=self.config.n_splits,
                    **cv_metrics
                )

                self.results.append(cv_result)

            # --------------------------------------------------
            # Imbalance Artifact
            # --------------------------------------------------

            if imbalance_summary:

                artifacts["imbalance"] = imbalance_summary

            elif class_dist_before:

                artifacts["imbalance"] = {
                    "method": None,
                    "before": class_dist_before,
                    "after": None
                }

            # --------------------------------------------------
            # Build Result
            # --------------------------------------------------

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                result_type="baseline",
                imbalance_summary=imbalance_summary,
                artifacts=artifacts,
                **metrics
            )

            exp_id = result["experiment"]

            self.trained_models[exp_id] = {
                "wrapper": wrapper,
                "result": result
            }

        except Exception as e:

            Logger.error(
                f"Model {model_name} failed during experiment: {e}"
            )

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                result_type="failed",
                artifacts={
                    "error": str(e)
                }
            )

        self.results.append(result)

        return result

    # ----------------------------
    # Run All Models
    # ----------------------------

    def run_all_models(self):

        models = self.registry.get_models_by_task("classification")

        results = []

        for name in models.keys():
            try:
                results.append(self.run_experiment(name))
            except Exception as e:
                Logger.error(f"Experiment for model {name} failed: {e}")

        return pd.DataFrame(results)
    # ---------------------------------------------------
    # TUNING
    # ---------------------------------------------------

    def tune_model(self, model_name, param_config=None, search_type="grid", cv=5, n_iter=20, **kwargs):
        try:
            if self.X_train is None:
                raise ValueError("Call prepare_data() before tuning")

            # ✅ Get model
            wrapper = copy.deepcopy(self.registry.get_model(model_name))

            # ✅ Inject SMOTE
            smote_handler = self._get_imbalance_handler()
            if smote_handler:
                wrapper.set_imbalance_handler(smote_handler)

            # ✅ Build pipeline
            wrapper.build_pipeline(self.preprocessor)

            # ✅ Build param_config if not provided
            if param_config is None:
                if not kwargs:
                    raise ValueError(
                        "Either param_config or model parameters (**kwargs) must be provided"
                    )

                param_config = {
                    f"model__{k}": v if isinstance(v, list) else [v]
                    for k, v in kwargs.items()
                }

            # ✅ Initialize tuner
            tuner = ClassificationHyperparameterTuner(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test
            )

            # ✅ Run tuning
            raw_results = tuner.tune(
                wrapper=wrapper,
                model_name=model_name,
                search_type=search_type,
                param_config=param_config,
                cv=cv,
                n_iter=n_iter
            )

            final_results = []

            # ✅ Extract imbalance summary (same for all runs)
            imbalance_summary = None
            if wrapper.imbalance_handler:
                imbalance_summary = wrapper.imbalance_handler.get_summary()

            # ✅ Normalize using ResultBuilder ✅
            for i, row in enumerate(raw_results):
                exp_id = f"{model_name} | {search_type} | run_{i}"
                artifacts = row.get("artifacts", {})

                # ✅ Remove artifacts from metrics dict (clean separation)
                metrics = {k: v for k, v in row.items() if k != "artifacts"}

                result = ResultBuilder.build(
                    model=model_name,
                    family=getattr(wrapper, "family", "unknown"),
                    experiment=exp_id,
                    mode="cv",
                    result_type="tuned",
                    imbalance_summary=imbalance_summary,
                    artifacts=artifacts,

                    # ✅ Tuning metadata
                    extra={
                        "search_type": search_type,
                        "best_params": row.get("best_params"),
                        "cv": cv,
                        "n_iter": n_iter
                    },

                    **metrics
                )

                best_pipeline = row.get("best_estimator") or row.get("pipeline")

                if best_pipeline is not None:
                    self.trained_models[exp_id] = {
                        "pipeline": best_pipeline,
                        "result": result
                    }

                final_results.append(result)

            # ✅ Store results
            self.results.extend(final_results)

            return final_results
        except Exception as e:
            Logger.error(f"ClassificationModelUtility.tune_model failed for {model_name}: {e}")
            raise

    # ---------------------------------------------------
    # TUNE ALL MODELS
    # ---------------------------------------------------

    def tune_all_models(
        self,
        param_configs,
        search_type="grid",
        cv=5,
        n_iter=20,
        **kwargs
    ):

        all_results = []

        for model_name in param_configs:

            Logger.info(f"🔧 Tuning {model_name}...")

            res = self.tune_model(
                model_name=model_name,
                param_config=param_configs[model_name],
                search_type=search_type,
                cv=cv,
                n_iter=n_iter,
                **kwargs
            )

            all_results.extend(res)

        return pd.DataFrame(all_results)

    # ---------------------------------------------------
    # RESULTS UTILITIES
    # ---------------------------------------------------

    def get_results_df(self):

        df = pd.DataFrame(self.results)

        # ❌ remove artifacts column (optional)
        if "artifacts" in df.columns:
            df = df.drop(columns=["artifacts"])

        return df

    def get_artifacts_df(self):

        rows = []

        for r in self.results:

            artifacts = r.get("artifacts", {})

            row = {
                "model": r["model"]
            }

            # ✅ Confusion Matrix → string/table friendly
            if "confusion_matrix" in artifacts:
                cm = artifacts["confusion_matrix"]
                row["confusion_matrix"] = str(cm.tolist())

            # ✅ ROC AUC summary only (optional)
            if "roc_curve" in artifacts:
                row["has_roc_curve"] = True
                row["roc_curve"] = artifacts["roc_curve"]
                row["roc_auc"] = r.get("roc_auc")

            else:
                row["has_roc_curve"] = False

            if "imbalance" in artifacts:
                imb = artifacts["imbalance"]
                row["imbalance_method"] = imb.get("method")

                # optional summary
                if imb.get("before") and imb.get("after"):
                    row["imbalance_changed"] = (
                        str(imb["before"]) + " → " + str(imb["after"])
                    )

            rows.append(row)

        return pd.DataFrame(rows)

    # ---------------------------------------------------
    # RANK MODELS
    # ---------------------------------------------------

    def rank_models(self, metric="accuracy", ascending=False):
        return ClassificationModelComparator(self.results).rank(metric, ascending)

    # ---------------------------------------------------
    # BEST MODEL
    # ---------------------------------------------------

    def get_best_model(self, metric="accuracy"):

        best = ClassificationModelComparator(self.results).best_model(metric)

        if best is None:
            return None

        # ✅ remove artifacts before returning
        clean_best = {k: v for k, v in best.items() if k != "artifacts"}

        return clean_best

    # ---------------------------------------------------
    # COMPARE MODELS
    # ---------------------------------------------------

    def compare_models(self):
        return ClassificationModelComparator(self.results).compare()

    def get_confusion_matrix_df(self, model_name):

        result = next((r for r in self.results if r["model"] == model_name), None)

        if not result:
            return None

        cm = result.get("artifacts", {}).get("confusion_matrix")

        return cf.confusion_matrix(cm)

    # ---------------------------------------------------
    # CONFUSION MATRICES
    # ---------------------------------------------------
    def get_all_confusion_matrices(self):

        cm_dict = {}

        for r in self.results:

            cm = r.get("artifacts", {}).get("confusion_matrix")

            if cm is not None:
                cm_dict[r["model"]] = cf.confusion_matrix(cm)

        return cm_dict

    # ---------------------------------------------------
    # ARTIFACT EXTRACTION
    # ---------------------------------------------------
    def _extract_artifacts(self, metrics):

        artifact_keys = {"roc_curve", "pr_curve", "classification_report"}

        if self.problem_type != "multilabel-indicator":
            artifact_keys.add("confusion_matrix")

        artifacts = {}
        numeric_metrics = {}

        for key, val in metrics.items():
            if key in artifact_keys and val is not None:
                artifacts[key] = val
            else:
                numeric_metrics[key] = val

        return artifacts, numeric_metrics

    # ---------------------------------------------------
    # ENSEMBLE MODELS
    # ---------------------------------------------------

    def run_ensemble(self, config):
        try:
            method = config.get("method")

            # ✅ Resolve models (parallel only)
            if config["type"] == "parallel":
                resolved_models = []
                for name in config.get("model_names", []):
                    base_wrapper = self.registry.get_model(name)
                    resolved_models.append((name, base_wrapper.model))

                wrapper = ParallelEnsembleWrapper(
                    models=resolved_models,
                    method=method,
                    voting=config.get("voting", "soft")
                )

                model_name = f"Ensemble_{method}_Voting"

            elif config["type"] == "sequential":
                wrapper = SequentialEnsembleWrapper(
                    method=method,
                    model=config.get("model")
                )

                model_name = f"Ensemble_{method}"

            elif config["type"] == "stacking":
                resolved_models = []
                for name in config.get("model_names", []):
                    base_wrapper = self.registry.get_model(name)
                    resolved_models.append((name, base_wrapper.model))

                # ✅ Resolve meta model from registry
                meta_name = config.get("meta_model")

                meta_wrapper = self.registry.get_model(meta_name)
                final_estimator = meta_wrapper.model

                wrapper = StackingEnsembleWrapper(
                    models=resolved_models,
                    final_estimator=final_estimator
                )

                model_name = f"Ensemble_Stacking_{meta_name}"

            else:
                raise ValueError("Invalid ensemble type")

            # ✅ Inject SMOTE
            handler = self._get_imbalance_handler()
            if handler:
                wrapper.set_imbalance_handler(handler)

            # ✅ Build + Train
            wrapper.build_pipeline(self.preprocessor)
            wrapper.train(self.X_train, self.y_train)

            # ✅ Prediction
            y_pred = wrapper.predict(self.X_test)
            y_proba = wrapper.predict_proba(self.X_test)

            # ✅ Evaluation
            metrics = wrapper.evaluate(self.y_test, y_pred, y_proba)
            artifacts, metrics = self._extract_artifacts(metrics)

            imbalance_summary = None
            if wrapper.imbalance_handler:
                imbalance_summary = wrapper.imbalance_handler.get_summary()

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "ensemble"),
                result_type="ensemble",
                mode="train-test",
                imbalance_summary=imbalance_summary,
                artifacts=artifacts,
                # ✅ Ensemble metadata
                extra={
                    "ensemble_type": config["type"],
                    "method": method,
                    "base_models": config.get("model_names")
                },
                **metrics
            )

            self.results.append(result)

            # ✅ STORE
            exp_id = result["experiment"]
            self.trained_models[exp_id] = {
                "wrapper": wrapper,
                "result": result
            }

            return result
        except Exception as e:
            Logger.error(f"ClassificationModelUtility.run_ensemble failed: {e}")
            raise

    # ======================================================
    # Plot Data Extraction for Visualization
    # ======================================================
    def get_plot_data(self):

        plot_rows = []

        for r in self.results:

            artifacts = r.get("artifacts", {})

            row = {
                "model": r["model"],
                "roc_auc": r.get("roc_auc"),
                "pr_auc": r.get("pr_auc"),
            }

            if artifacts.get("roc_curve"):
                row["roc_curve"] = artifacts["roc_curve"]

            if artifacts.get("pr_curve"):
                row["pr_curve"] = artifacts["pr_curve"]

            # ✅ ADD THIS BLOCK (IMPORTANT)
            if "best_threshold" in r:
                row["best_threshold"] = r.get("best_threshold")
                row["best_fpr"] = r.get("best_fpr")
                row["best_tpr"] = r.get("best_tpr")

            plot_rows.append(row)

        return plot_rows

    # ---------------------------------------------------
    # MODEL PERSISTENCE
    # ---------------------------------------------------
    def save_model(self, exp_id, path):
        try:
            if exp_id not in self.trained_models:
                raise ValueError(f"{exp_id} not found")

            model_obj = self.trained_models[exp_id]

            # ✅ Support both wrapper and pipeline
            if "pipeline" in model_obj:
                pipeline = model_obj["pipeline"]
                result = model_obj["result"]
            else:
                wrapper = model_obj["wrapper"]
                pipeline = wrapper.get_pipeline()
                result = model_obj["result"]

            os.makedirs(path, exist_ok=True)

            joblib.dump(pipeline, f"{path}/pipeline.pkl")

            metadata = {
                "model": result.get("model"),
                "task": result.get("task"),
                "family": result.get("family"),
                "experiment": exp_id,
                "feature_names": list(self.X_train.columns),
                "inference_version": "v1",
                "pipeline_type": "sklearn_pipeline",
                "validated": False,
                "extra": result.get("extra", {}),
                "problem_type": result.get("problem_type"),
                "threshold": result.get("best_threshold"),
            }

            with open(f"{path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            Logger.info(f"✅ Model saved: {exp_id}")
        except Exception as e:
            Logger.error(f"ClassificationModelUtility.save_model failed for {exp_id}: {e}")
            raise

    def validate_inference_pipeline(self, exp_id, model_path,
                                    atol=1e-6, rtol=1e-5, validate_proba=True, validate_threshold=True):
        """
        ✅ Validate inference pipeline against trained model

        Supports:
        - binary
        - multiclass
        - multilabel

        Checks:
        - predictions
        - probabilities (optional)
        - threshold predictions (optional)
        """

        try:
            if exp_id not in self.trained_models:
                raise ValueError(f"{exp_id} not found in trained models")

            model_obj = self.trained_models[exp_id]
            wrapper = None
            pipeline = None

            # ✅ ----------------------------------
            # 1. TRAINING predictions
            # ✅ ----------------------------------
            train_proba = None
            if "wrapper" in model_obj:
                wrapper = model_obj["wrapper"]
                train_preds = wrapper.predict(self.X_test)
                if validate_proba and hasattr(wrapper, "predict_proba"):
                    try:
                        train_proba = wrapper.predict_proba(self.X_test)
                    except BaseException:
                        pass
            else:
                pipeline = model_obj["pipeline"]
                train_preds = pipeline.predict(self.X_test)
                if validate_proba and hasattr(pipeline, "predict_proba"):
                    try:
                        train_proba = pipeline.predict_proba(self.X_test)
                    except BaseException:
                        pass

            # ✅ ----------------------------------
            # 2. INFERENCE predictions
            # ✅ ----------------------------------
            inf_model = InferenceFactory.load(model_path)
            inf_preds = inf_model.predict(self.X_test)

            inf_proba = None
            if validate_proba and hasattr(inf_model, "predict_proba"):
                try:
                    inf_proba = inf_model.predict_proba(self.X_test)
                except BaseException:
                    pass

            inf_threshold_preds = None
            if validate_threshold and hasattr(inf_model, "predict_with_threshold"):
                try:
                    inf_threshold_preds = inf_model.predict_with_threshold(self.X_test)
                except BaseException:
                    pass

            # ✅ ----------------------------------
            # 3. VALIDATION
            # ✅ ----------------------------------
            result = {
                "exp_id": exp_id,
                "sample_size": len(train_preds)
            }

            preds_match = np.array_equal(train_preds, inf_preds)
            result["predictions_match"] = preds_match

            proba_match = None
            if train_proba is not None and inf_proba is not None:
                proba_match = np.allclose(train_proba, inf_proba, atol=atol, rtol=rtol)
                result["proba_match"] = proba_match

            threshold_match = None
            if inf_threshold_preds is not None:
                train_threshold_preds = None
                if wrapper is not None and hasattr(wrapper, "predict_with_threshold"):
                    try:
                        train_threshold_preds = wrapper.predict_with_threshold(self.X_test)
                    except BaseException:
                        pass

                if train_threshold_preds is not None:
                    threshold_match = np.array_equal(train_threshold_preds, inf_threshold_preds)
                result["threshold_match"] = threshold_match

            is_valid = (
                preds_match
                and (proba_match if proba_match is not None else True)
                and (threshold_match if threshold_match is not None else True)
            )
            result["status"] = "PASS" if is_valid else "FAIL"

            if not is_valid:
                diff = None
                try:
                    diff = np.abs(train_preds - inf_preds)
                except BaseException:
                    pass

                result.update({
                    "max_diff": float(np.max(diff)) if diff is not None else None,
                    "mismatch_indices": (
                        list(np.where(train_preds != inf_preds)[0][:10])
                        if not preds_match else []
                    )
                })

            return result
        except Exception as e:
            Logger.error(f"ClassificationModelUtility.validate_inference_pipeline failed for {exp_id}: {e}")
            raise
