import pandas as pd
from sklearn.model_selection import train_test_split

from lib.utility.machinelearning.evaluation.ClassificationModelComparator import ClassificationModelComparator
from lib.utility.machinelearning.evaluation.Metrics import Metrics
from lib.utility.machinelearning.experiment.ExperimentRunner import ExperimentRunner
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
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

    # ----------------------------
    def prepare_data(self, test_size=0.2, random_state=42):

        df = self.df.copy()

        if self.imputer:
            df = self.imputer.fit_transform(df)

        if self.outlier_handler:
            df = self.outlier_handler.fit_transform(df)

        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
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

        wrapper.build_pipeline(self.preprocessor)
        wrapper.train(self.X_train, self.y_train)

        # ✅ predictions
        y_pred = wrapper.predict(self.X_test)

        # ✅ probabilities (IMPORTANT)
        y_proba = None
        if hasattr(wrapper.pipeline, "predict_proba"):
            try:
                y_proba = wrapper.pipeline.predict_proba(self.X_test)
            except Exception:
                y_proba = None

        # ✅ FULL metrics
        metrics = Metrics.classification(
            self.y_test,
            y_pred,
            y_proba=y_proba,
            include_confusion_matrix=True,
            include_curves=True   # ✅ enables ROC + PR
        )

        result = {
            "model": model_name,
            "experiment": f"{model_name} | classification",
            "mode": "train-test",
            "type": "baseline",
            **metrics
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
        return pd.DataFrame(self.results)

    # ---------------------------------------------------
    # RANK MODELS
    # ---------------------------------------------------

    def rank_models(self, metric="accuracy", ascending=False):
        return ClassificationModelComparator(self.results).rank(metric, ascending)

    # ---------------------------------------------------
    # BEST MODEL
    # ---------------------------------------------------

    def get_best_model(self, metric="accuracy"):
        return ClassificationModelComparator(self.results).best_model(metric)

    # ---------------------------------------------------
    # COMPARE MODELS
    # ---------------------------------------------------

    def compare_models(self):
        return ClassificationModelComparator(self.results).compare()
