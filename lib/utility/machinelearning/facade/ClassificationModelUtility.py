import copy
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from skmultilearn.model_selection import iterative_train_test_split

from lib.utility.logger import Logger
from lib.utility.machinelearning.base.ParallelEnsembleWrapper import ParallelEnsembleWrapper
from lib.utility.machinelearning.base.SequentialEnsembleWrapper import SequentialEnsembleWrapper
from lib.utility.machinelearning.base.StackingEnsembleWrapper import StackingEnsembleWrapper
from lib.utility.machinelearning.evaluation.ClassificationModelComparator import ClassificationModelComparator
from lib.utility.machinelearning.pipeline.imbalance.SMOTEHandler import SMOTEHandler
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.ClassificationFormatter import ClassificationFormatter as cf
from lib.utility.machinelearning.shared.ResultBuilder import ResultBuilder
from lib.utility.machinelearning.tuning.ClassificationHyperparameterTuner import ClassificationHyperparameterTuner


class ClassificationModelUtility:

    def __init__(self, df, target_col, imputer=None, outlier_handler=None, imbalance_config=None):

        self.df = df
        self.target_col = target_col

        self.imputer = imputer
        self.outlier_handler = outlier_handler
        self.imbalance_config = imbalance_config

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.results = []

        self.registry = ModelRegistry()
        self.preprocessor = None
        self.problem_type = None

    # ----------------------------
    # Data Preparation
    # ----------------------------
    def prepare_data(self, test_size=0.2, random_state=42):

        df = self.df.copy()

        if self.imputer:
            df = self.imputer.fit_transform(df)

        if self.outlier_handler:
            df = self.outlier_handler.fit_transform(df)

        X = df.drop(self.target_col, axis=1)

        y = df[self.target_col]

        # ✅ FIX: encode string labels → numeric

        self.label_encoder = None

        if isinstance(y, pd.Series) and not pd.api.types.is_numeric_dtype(y):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        self.problem_type = type_of_target(y)
        Logger.info(f"Detected problem type: {self.problem_type}")

        if self.problem_type == "multilabel-indicator":

            X_np = X.values
            y_np = y.values

            X_train, y_train, X_test, y_test = iterative_train_test_split(
                X_np, y_np, test_size=test_size
            )

            self.X_train = pd.DataFrame(X_train, columns=X.columns)
            self.X_test = pd.DataFrame(X_test, columns=X.columns)
            self.y_train = pd.DataFrame(y_train, columns=y.columns)
            self.y_test = pd.DataFrame(y_test, columns=y.columns)

        else:

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        self.preprocessor = Preprocessor(
            self.X_train,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler
        ).build()

    # ----------------------------
    # Imbalance Handling
    # ----------------------------

    def _get_imbalance_handler(self):

        if not self.imbalance_config:
            return None

        im_type = self.imbalance_config.get("type")

        if im_type == "smote":
            return SMOTEHandler(**self.imbalance_config.get("params", {}))

        return None

    # ----------------------------
    # Run Single Experiment
    # ----------------------------
    def run_experiment(self, model_name):

        wrapper = copy.deepcopy(self.registry.get_model(model_name))

        try:
            if self.problem_type == "multilabel-indicator":
                wrapper.model = OneVsRestClassifier(wrapper.model)

            # ✅ Inject SMOTE

            smote_handler = self._get_imbalance_handler()

            if smote_handler:
                wrapper.set_imbalance_handler(smote_handler)

            # ✅ Build pipeline (P → P1 → P2 → P3)
            wrapper.build_pipeline(self.preprocessor)

            # ✅ TRAIN
            wrapper.train(self.X_train, self.y_train)

            # ✅ Capture BEFORE distribution
            class_dist_before = dict(Counter(self.y_train))

            # ✅ Extract SMOTE summary after training
            imbalance_summary = None

            if wrapper.imbalance_handler:
                imbalance_summary = wrapper.imbalance_handler.get_summary()

            # ✅ PREDICT (only if train succeeded)
            y_pred = wrapper.predict(self.X_test)

            y_proba = wrapper.predict_proba(self.X_test)

            metrics = wrapper.evaluate(self.y_test, y_pred, y_proba)

            artifacts, metrics = self._extract_artifacts(metrics)

            # ✅ Add imbalance info to artifacts
            if imbalance_summary:
                artifacts["imbalance"] = imbalance_summary
            else:
                artifacts["imbalance"] = {
                    "method": None,
                    "before": class_dist_before,
                    "after": None
                }

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                result_type="baseline",
                imbalance_summary=imbalance_summary,
                artifacts=artifacts,
                **metrics
            )

        except Exception as e:

            Logger.error(f"Model {model_name} failed during experiment: {e}")

            # ✅ SAFE FAILURE RECORD
            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                result_type="failed",
                artifacts={"error": str(e)}
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
        pipeline = wrapper.get_pipeline()

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
            pipeline=pipeline,
            model_name=model_name,
            search_type=search_type,
            param_config=param_config,
            cv=cv,
            n_iter=n_iter
        )

        exp_name = f"{model_name} | {search_type}"

        final_results = []

        # ✅ Extract imbalance summary (same for all runs)
        imbalance_summary = None
        if wrapper.imbalance_handler:
            imbalance_summary = wrapper.imbalance_handler.get_summary()

        # ✅ Normalize using ResultBuilder ✅
        for row in raw_results:

            artifacts = row.get("artifacts", {})

            # ✅ Remove artifacts from metrics dict (clean separation)
            metrics = {k: v for k, v in row.items() if k != "artifacts"}

            result = ResultBuilder.build(
                model=model_name,
                family=getattr(wrapper, "family", "unknown"),
                experiment=exp_name,
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

            final_results.append(result)

        # ✅ Store results
        self.results.extend(final_results)

        return final_results

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

        method = config.get("method")

        resolved_models = []

        # ✅ Resolve models (parallel only)
        if config["type"] == "parallel":

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

        return result
