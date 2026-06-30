import copy
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from lib.utility.logger import Logger
from lib.utility.machinelearning.evaluation.ModelComparator import ModelComparator
from lib.utility.machinelearning.inference.InferenceFactory import InferenceFactory
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.Formatter import Formatter
from lib.utility.machinelearning.tuning.HyperparameterTuner import HyperparameterTuner


class LinearModelUtility:

    def __init__(self, X_train, y_train, X_test=None, y_test=None,
                 imputer=None, outlier_handler=None):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.imputer = imputer
        self.outlier_handler = outlier_handler

        self.preprocessor = None
        self.results = []
        self.trained_models = {}
        self.registry = ModelRegistry()

    # ---------------------------------------------------
    # DATA PREPARATION
    # ---------------------------------------------------

    def prepare_data(self):
        try:
            self.preprocessor = Preprocessor(
                self.X_train,
                imputer=self.imputer,
                outlier_handler=self.outlier_handler
            ).build()
        except Exception as e:
            Logger.error(f"LinearModelUtility.prepare_data failed: {e}")
            raise

    # ---------------------------------------------------
    # RUN SINGLE EXPERIMENT
    # ---------------------------------------------------
    def run_experiment(self, model_name, k_fold=None, imputer=None, outlier_handler=None):
        try:
            wrapper = copy.deepcopy(self.registry.get_model(model_name))

            # ✅ preprocessing override
            if imputer or outlier_handler:
                preprocessor = Preprocessor(
                    self.X_train,
                    imputer=imputer or self.imputer,
                    outlier_handler=outlier_handler or self.outlier_handler
                ).build()
            else:
                preprocessor = self.preprocessor

            wrapper.build_pipeline(preprocessor)

            mode = "k-fold" if k_fold else "train-test"
            exp_name = Formatter.build(model_name, mode, k_fold, imputer, outlier_handler)
            Logger.info(f"Running experiment: {exp_name}")

            # ✅ -------------------------
            # K-FOLD CASE
            # ✅ -------------------------
            if k_fold:

                kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
                pipeline = wrapper.get_pipeline()

                # ✅ CV predictions (evaluation only)
                y_pred = cross_val_predict(
                    pipeline,
                    self.X_train,
                    self.y_train,
                    cv=kf,
                    n_jobs=8
                )

                metrics = wrapper.evaluate(self.y_train, y_pred)

                # ✅ TRAIN FINAL MODEL ON FULL DATA (MANDATORY)
                wrapper.train(self.X_train, self.y_train)

            # ✅ -------------------------
            # TRAIN-TEST CASE
            # ✅ -------------------------
            else:

                wrapper.train(self.X_train, self.y_train)
                y_pred = wrapper.predict(self.X_test)
                metrics = wrapper.evaluate(self.y_test, y_pred)

            # ✅ artifact extraction
            artifacts, metrics = self._extract_artifacts(metrics)

            # ✅ BUILD RESULT
            result = {
                "model": model_name,
                "task": getattr(wrapper, "task", "regression"),
                "family": getattr(wrapper, "family", "unknown"),
                "experiment": exp_name,
                "mode": mode,
                "type": "baseline",
                "k": k_fold if k_fold else None,
                **metrics,
                "artifacts": artifacts
            }

            # ✅ ✅ STORE WRAPPER + RESULT TOGETHER (CRITICAL CHANGE)
            self.trained_models[exp_name] = {
                "wrapper": wrapper,
                "result": result
            }

            self.results.append(result)
            return result
        except Exception as e:
            Logger.error(f"LinearModelUtility.run_experiment failed for {model_name}: {e}")
            raise
    # ---------------------------------------------------
    # RUN ALL MODELS
    # ---------------------------------------------------

    def run_all_models(self, k_fold=None):

        models = self.registry.get_models_by_task("linear regression")

        results = []

        for model_name in models.keys():

            Logger.info(f"Running model: {model_name}")

            try:
                result = self.run_experiment(model_name, k_fold=k_fold)
                results.append(result)

            except Exception as e:

                Logger.error(f"Critical failure in {model_name}: {str(e)}")

                results.append({
                    "model": model_name,
                    "task": "regression",
                    "type": "failed",
                    "error": str(e)
                })

        Logger.info(f"Completed {len(results)} experiments")

        return pd.DataFrame(results)
    # ---------------------------------------------------
    # RUN CUSTOM CONFIGS
    # ---------------------------------------------------

    def run_experiments(self, configs):

        all_results = []

        for config in configs:

            model_name = config.get("model_name")
            k_fold = config.get("k_fold", None)

            try:
                wrapper = copy.deepcopy(self.registry.get_model(model_name))

                wrapper.build_pipeline(self.preprocessor)

                mode = "k-fold" if k_fold else "train-test"

                exp_name = Formatter.build(
                    model_name=model_name,
                    mode=mode,
                    k=config.get("k_fold"),
                    imputer=config.get("imputer"),
                    outlier_handler=config.get("outlier_handler"),
                    search_type="custom"
                )

                # ✅ K-FOLD
                if k_fold:

                    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
                    pipeline = wrapper.get_pipeline()

                    y_pred = cross_val_predict(
                        pipeline,
                        self.X_train,
                        self.y_train,
                        cv=kf,
                        n_jobs=8
                    )

                    metrics = wrapper.evaluate(self.y_train, y_pred)

                    # ✅ TRAIN FINAL MODEL ON FULL DATA (MANDATORY)
                    wrapper.train(self.X_train, self.y_train)

                # ✅ TRAIN-TEST
                else:
                    wrapper.train(self.X_train, self.y_train)

                    y_pred = wrapper.predict(self.X_test)

                    metrics = wrapper.evaluate(self.y_test, y_pred)

                artifacts, metrics = self._extract_artifacts(metrics)

                result = {
                    "model": model_name,
                    "task": getattr(wrapper, "task", "regression"),
                    "family": getattr(wrapper, "family", "unknown"),
                    "experiment": exp_name,
                    "mode": mode,
                    "type": "custom",
                    **metrics,
                    "artifacts": artifacts
                }

                # STORE WRAPPER + RESULT TOGETHER (CRITICAL CHANGE)
                self.trained_models[exp_name] = {
                    "wrapper": wrapper,
                    "result": result
                }

            except Exception as e:
                result = {
                    "model": model_name,
                    "experiment": f"{model_name} | custom",
                    "type": "failed",
                    "error": str(e)
                }
                Logger.error(f"Experiment failed for {model_name}: {str(e)}")

            self.results.append(result)
            all_results.append(result)

        return pd.DataFrame(all_results)

    # ---------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------
    def grid_search_cv(
        self,
        model_name,
        param_grid,
        cv=5,
        scoring=None
    ):

        try:
            wrapper = copy.deepcopy(self.registry.get_model(model_name))
            wrapper.build_pipeline(self.preprocessor)

            pipeline = wrapper.get_pipeline()

            scoring = scoring or "neg_mean_squared_error"

            Logger.info(f"Starting GridSearchCV for {model_name} with params: {param_grid} and cv={cv}")

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )

            # ✅ TRAIN
            grid.fit(self.X_train, self.y_train)

            best_model = grid.best_estimator_

            exp_id = f"{model_name} | gridsearch"

            y_pred = best_model.predict(self.X_test)

            # ✅ evaluate using wrapper
            metrics = wrapper.evaluate(self.y_test, y_pred)

            artifacts, metrics = self._extract_artifacts(metrics)

            result = {
                "model": model_name,
                "task": getattr(wrapper, "task", "regression"),
                "family": getattr(wrapper, "family", "unknown"),
                "experiment": f"{model_name} | gridsearch",
                "type": "tuned",
                "mode": "gridsearch",
                "search_type": "grid",
                "best_params": grid.best_params_,
                "best_score_cv": grid.best_score_,
                **metrics,
                "artifacts": artifacts
            }

            self.trained_models[exp_id] = {
                "pipeline": best_model,
                "result": result
            }

        except Exception as e:

            result = {
                "model": model_name,
                "experiment": f"{model_name} | gridsearch",
                "type": "failed",
                "error": str(e)
            }

        self.results.append(result)
        return result
    # ---------------------------------------------------
    # TUNING
    # ---------------------------------------------------

    def tune_model(self, model_name, param_grid, search_type="grid", cv=5, n_iter=20):
        try:
            wrapper = copy.deepcopy(self.registry.get_model(model_name))
            wrapper.build_pipeline(self.preprocessor)

            tuner = HyperparameterTuner(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test
            )

            # ✅ Run tuning
            if search_type == "grid":
                results = tuner.grid_search(
                    wrapper=wrapper,
                    model_name=model_name,
                    param_grid=param_grid,
                    cv=cv
                )

            elif search_type == "random":
                results = tuner.random_search(
                    wrapper=wrapper,
                    model_name=model_name,
                    param_dist=param_grid,
                    n_iter=n_iter,
                    cv=cv
                )

            else:
                raise ValueError(f"Unsupported search_type: {search_type}")

            final_results = []

            # ✅ Process each result row
            for i, row in enumerate(results):

                exp_name = f"{model_name} | {search_type} | run_{i}"

                # ✅ enrich result
                row.update({
                    "model": model_name,
                    "task": getattr(wrapper, "task", "regression"),
                    "family": getattr(wrapper, "family", "unknown"),
                    "type": row.get("type", "tuned"),
                    "search_type": search_type,
                    "experiment": exp_name
                })

                best_pipeline = row.get("best_estimator") or row.get("pipeline")

                if best_pipeline is not None:
                    self.trained_models[exp_name] = {
                        "pipeline": best_pipeline,
                        "result": row
                    }

                final_results.append(row)

            # ✅ persist results
            self.results.extend(final_results)

            return final_results
        except Exception as e:
            Logger.error(f"LinearModelUtility.tune_model failed for {model_name}: {e}")
            raise

    # ---------------------------------------------------
    # RESULT UTILITIES
    # ---------------------------------------------------

    def get_results_df(self):
        df = pd.DataFrame(self.results)
        return df.drop(columns=["artifacts"], errors="ignore")

    def rank_models(self, metric=None, ascending=None):
        comparator = ModelComparator.get_comparator(self.results)
        return comparator.rank(metric, ascending)

    def get_best_model(self, metric=None):
        comparator = ModelComparator.get_comparator(self.results)

        best = comparator.best_model(metric)
        if best is None:
            return None

        return {k: v for k, v in best.items() if k != "artifacts"}

    def compare_models(self, metric=None):
        comparator = ModelComparator.get_comparator(self.results)
        return comparator.compare(metric)
    # ---------------------------------------------------
    # INTERNAL
    # ---------------------------------------------------

    def _extract_artifacts(self, metrics):

        # ✅ future extension (residuals, plots etc.)
        artifact_keys = {"residuals", "prediction_vs_actual"}

        artifacts = {}
        numeric_metrics = {}

        for key, val in metrics.items():
            if key in artifact_keys and val is not None:
                artifacts[key] = val
            else:
                numeric_metrics[key] = val

        return artifacts, numeric_metrics

    # ---------------------------------------------------
    # MODEL PERSISTENCE
    # ---------------------------------------------------

    def save_model(self, exp_id, path):
        try:
            if exp_id not in self.trained_models:
                raise ValueError(f"{exp_id} not found")

            model_obj = self.trained_models[exp_id]

            # ✅ Support both cases
            if "pipeline" in model_obj:
                pipeline = model_obj["pipeline"]   # ✅ tuned/grid case
            else:
                wrapper = model_obj["wrapper"]
                pipeline = wrapper.get_pipeline()  # ✅ baseline case

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
                "target_mean": float(self.y_train.mean())
            }

            with open(f"{path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            Logger.info(f"✅ Model saved: {exp_id}")
        except Exception as e:
            Logger.error(f"LinearModelUtility.save_model failed for {exp_id}: {e}")
            raise

    def validate_inference_pipeline(self, exp_id, model_path, atol=1e-6, rtol=1e-5):
        """
        ✅ Validates that saved inference pipeline produces identical predictions
        as the trained model pipeline.

        Parameters
        ----------
        exp_id : str
            Experiment ID used during training
        model_path : str
            Path where model is saved (contains pipeline.pkl)
        atol : float
            Absolute tolerance for comparison
        rtol : float
            Relative tolerance for comparison

        Returns
        -------
        dict
            Validation status and diagnostics
        """

        try:
            if exp_id not in self.trained_models:
                raise ValueError(f"{exp_id} not found in trained models")

            model_obj = self.trained_models[exp_id]

        # ✅ ----------------------------------
        # 1. Get TRAINING predictions
        # ✅ ----------------------------------
            if "wrapper" in model_obj:
                wrapper = model_obj["wrapper"]
                train_preds = wrapper.predict(self.X_test)
            elif "pipeline" in model_obj:
                pipeline = model_obj["pipeline"]
                train_preds = pipeline.predict(self.X_test)
            else:
                raise ValueError("Invalid model object structure")

        # ✅ ----------------------------------
        # 2. Get INFERENCE predictions
        # ✅ ----------------------------------
            inference_model = InferenceFactory.load(model_path)
            inf_preds = inference_model.predict(self.X_test)

        # ✅ ----------------------------------
        # 3. Compare predictions
        # ✅ ----------------------------------
            is_match = np.allclose(train_preds, inf_preds, atol=atol, rtol=rtol)

            result = {
                "status": "PASS" if is_match else "FAIL",
                "exp_id": exp_id,
                "sample_size": len(train_preds)
            }

        # ✅ ----------------------------------
        # 4. Diagnostics if FAIL
        # ✅ ----------------------------------
            if not is_match:
                diff = np.abs(train_preds - inf_preds)

                result.update({
                    "max_diff": float(np.max(diff)),
                    "mean_diff": float(np.mean(diff)),
                    "sample_mismatch_indices": list(np.where(diff > atol)[0][:10])
                })

            return result
        except Exception as e:
            Logger.error(f"LinearModelUtility.validate_inference_pipeline failed for {exp_id}: {e}")
            raise
