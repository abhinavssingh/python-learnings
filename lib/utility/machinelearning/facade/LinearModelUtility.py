import copy

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split

from lib.utility.logger import Logger
from lib.utility.machinelearning.evaluation.ModelComparator import ModelComparator
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.shared.Formatter import Formatter
from lib.utility.machinelearning.tuning.HyperparameterTuner import HyperparameterTuner


class LinearModelUtility:

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
        self.preprocessor = None
        self.task = "regression"

    # ---------------------------------------------------
    # DATA PREPARATION
    # ---------------------------------------------------
    def prepare_data(self, test_size=0.2, random_state=42):

        df = self.df.copy()

        if self.imputer:
            df = self.imputer.fit_transform(df)

        if self.outlier_handler:
            df = self.outlier_handler.fit_transform(df)

        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # ✅ FIX: use TRAIN DATA ONLY
        self.preprocessor = Preprocessor(
            self.X_train,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler
        ).build()

    # ---------------------------------------------------
    # RUN SINGLE EXPERIMENT
    # ---------------------------------------------------
    def run_experiment(self, model_name, k_fold=None, imputer=None, outlier_handler=None):

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

            y_pred = cross_val_predict(
                wrapper.get_pipeline(),
                self.X_train,
                self.y_train,
                cv=kf,
                n_jobs=-1
            )

            metrics = wrapper.evaluate(self.y_train, y_pred)

        # ✅ -------------------------
        # TRAIN-TEST CASE
        # ✅ -------------------------
        else:

            wrapper.train(self.X_train, self.y_train)

            y_pred = wrapper.predict(self.X_test)

            metrics = wrapper.evaluate(self.y_test, y_pred)

        # ✅ artifact extraction (future-ready)
        artifacts, metrics = self._extract_artifacts(metrics)

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

        self.results.append(result)
        return result

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
                    from sklearn.model_selection import KFold, cross_val_predict

                    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

                    y_pred = cross_val_predict(
                        wrapper.get_pipeline(),
                        self.X_train,
                        self.y_train,
                        cv=kf,
                        n_jobs=-1
                    )

                    metrics = wrapper.evaluate(self.y_train, y_pred)

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

            # ✅ 🚨 CRITICAL FIX: enforce regression scoring
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

        wrapper = copy.deepcopy(self.registry.get_model(model_name))
        wrapper.build_pipeline(self.preprocessor)

        tuner = HyperparameterTuner(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )

        # ✅ FIX: use wrapper-based API + keyword args (important)
        if search_type == "grid":
            results = tuner.grid_search(
                wrapper=wrapper, model_name=model_name, param_grid=param_grid, cv=cv)

        elif search_type == "random":
            results = tuner.random_search(wrapper=wrapper, model_name=model_name,
                                          param_dist=param_grid, n_iter=n_iter, cv=cv)

        else:
            raise ValueError(f"Unsupported search_type: {search_type}")

        # ✅ enrich results (minimal change from your version)
        for row in results:
            row.update({
                "model": model_name,
                "task": getattr(wrapper, "task", "regression"),
                "family": getattr(wrapper, "family", "unknown"),
                "type": row.get("type", "tuned"),
                "search_type": search_type
            })

        # ✅ persist
        self.results.extend(results)

        return results

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
