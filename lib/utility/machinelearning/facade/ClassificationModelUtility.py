import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import type_of_target
from skmultilearn.model_selection import iterative_train_test_split

from lib.utility.machinelearning.evaluation.ClassificationModelComparator import ClassificationModelComparator
from lib.utility.machinelearning.evaluation.Metrics import Metrics
from lib.utility.machinelearning.experiment.ExperimentRunner import ExperimentRunner
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.ClassificationFormatter import ClassificationFormatter as cf
from lib.utility.machinelearning.tuning.ClassificationHyperparameterTuner import ClassificationHyperparameterTuner


class ClassificationModelUtility:

    def __init__(self, df, target_col, imputer=None, outlier_handler=None):

        self.df = df
        self.target_col = target_col

        self.imputer = imputer
        self.outlier_handler = outlier_handler

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.results = []

        self.registry = ModelRegistry()
        self.runner = None
        self.preprocessor = None
        self.problem_type = None

    # ----------------------------
    def prepare_data(self, test_size=0.2, random_state=42):

        df = self.df.copy()

        if self.imputer:
            df = self.imputer.fit_transform(df)

        if self.outlier_handler:
            df = self.outlier_handler.fit_transform(df)

        X = df.drop(self.target_col, axis=1)

        # ✅ support list of columns for multilabel classification
        if isinstance(self.target_col, list):
            y = df[self.target_col]
        else:
            y = df[self.target_col]

        # ✅ detect problem type
        self.problem_type = type_of_target(y)

        print(f"✅ Detected problem type: {self.problem_type}")

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
            X,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler
        ).build()

        self.runner = ExperimentRunner(self.preprocessor)

    # ----------------------------

    def run_experiment(self, model_name):

        if self.X_train is None:
            raise ValueError("Call prepare_data() before running experiments")

        wrapper = self.registry.get_model(model_name)

        # ✅ multilabel adaptation
        if self.problem_type == "multilabel-indicator":
            wrapper.model = OneVsRestClassifier(wrapper.model)

        wrapper.build_pipeline(self.preprocessor)

        # ✅ train + predict
        wrapper.train(self.X_train, self.y_train)
        y_pred = wrapper.predict(self.X_test)

        # ✅ probabilities
        y_proba = self._get_probabilities(wrapper)

        # ✅ compute metrics
        metrics = Metrics.classification(
            self.y_test,
            y_pred,
            y_proba=y_proba,
            include_confusion_matrix=(self.problem_type != "multilabel-indicator"),
            include_curves=True
        )

        # ✅ split artifacts (generic ✅)
        artifacts = self._extract_artifacts(metrics)

        result = {
            "model": model_name,
            "experiment": f"{model_name} | classification",
            "mode": "train-test",
            "type": "baseline",
            **metrics,
            "artifacts": artifacts
        }

        self.results.append(result)
        return result
    # ----------------------------

    def run_all_models(self):

        models = self.registry.get_models_by_task("classification")
        results = []

        for name in models.keys():
            results.append(self.run_experiment(name))

        return pd.DataFrame(results)

    # ---------------------------------------------------
    # TUNING
    # ---------------------------------------------------

    def tune_model(
        self,
        model_name,
        param_config=None,
        search_type="grid",
        cv=5,
        n_iter=20,
        **kwargs
    ):

        if self.X_train is None:
            raise ValueError("Call prepare_data() before tuning")

        # ✅ get model
        wrapper = self.registry.get_model(model_name)
        wrapper.build_pipeline(self.preprocessor)

        pipeline = wrapper.get_pipeline()

        # ✅ NEW: build param_config from kwargs if not provided
        if param_config is None:

            if not kwargs:
                raise ValueError(
                    "Either param_config or model parameters (**kwargs) must be provided"
                )

            param_config = {
                f"model__{k}": v if isinstance(v, list) else [v]
                for k, v in kwargs.items()
            }

        # ✅ initialize tuner
        tuner = ClassificationHyperparameterTuner(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )

        # ✅ call tuner
        results = tuner.tune(
            pipeline=pipeline,
            model_name=model_name,
            search_type=search_type,
            param_config=param_config,
            cv=cv,
            n_iter=n_iter
        )

        exp_name = f"{model_name} | {search_type}"

        # ✅ update rows
        for row in results:
            row.update({
                "model": model_name,
                "experiment": exp_name,
                "type": "tuned",
                "search_type": search_type
            })

        # ✅ store results
        self.results.extend(results)

        return results

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

            print(f"🔧 Tuning {model_name}...")

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

    def get_all_confusion_matrices(self):

        cm_dict = {}

        for r in self.results:

            cm = r.get("artifacts", {}).get("confusion_matrix")

            if cm is not None:
                cm_dict[r["model"]] = cf.confusion_matrix(cm)

        return cm_dict

    def _get_probabilities(self, wrapper):

        if not hasattr(wrapper.pipeline, "predict_proba"):
            return None

        try:
            raw_proba = wrapper.pipeline.predict_proba(self.X_test)

            if self.problem_type == "multilabel-indicator":

                if isinstance(raw_proba, list):
                    return np.column_stack([
                        p[:, 1] if p.ndim > 1 else p
                        for p in raw_proba
                    ])

            return raw_proba

        except Exception:
            return None

    def _extract_artifacts(self, metrics):

        artifact_keys = {"roc_curve", "pr_curve", "classification_report"}

        if self.problem_type != "multilabel-indicator":
            artifact_keys.add("confusion_matrix")

        artifacts = {}

        for key in list(metrics.keys()):
            if key in artifact_keys:
                artifacts[key] = metrics.pop(key)

        return artifacts
