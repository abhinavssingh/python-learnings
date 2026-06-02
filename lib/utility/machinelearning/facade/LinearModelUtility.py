import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from lib.utility.machinelearning.evaluation.ModelComparator import ModelComparator
from lib.utility.machinelearning.experiment.ExperimentRunner import ExperimentRunner
from lib.utility.machinelearning.pipeline.Preprocessor import Preprocessor
from lib.utility.machinelearning.registry.ModelRegistry import ModelRegistry
from lib.utility.machinelearning.tuning.HyperparameterTuner import HyperparameterTuner


class LinearModelUtility:
    """
    Facade layer for running ML experiments with loose coupling.
    """

    def __init__(self, df, target_col, imputer=None, outlier_handler=None):
        self.df = df
        self.target_col = target_col

        self.imputer = imputer
        self.outlier_handler = outlier_handler

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.experiment_results = []

        # ✅ decoupled components
        self.registry = ModelRegistry()
        self.runner = None
        self.preprocessor = None

    # ---------------------------------------------------
    # DATA PREPARATION
    # ---------------------------------------------------
    def prepare_data(self, test_size=0.2, random_state=42):

        df = self.df.copy()

        if self.imputer:
            df = self.imputer.fit_transform(df)

        if self.outlier_handler:
            df = self.outlier_handler.fit_transform(df)

        self.X = df.drop(self.target_col, axis=1)
        self.y = df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # ✅ FIX: pass imputer/outlier into preprocessor
        self.preprocessor = Preprocessor(
            self.X,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler
        ).build()

        self.runner = ExperimentRunner(self.preprocessor)

    # ---------------------------------------------------
    # RUN SINGLE EXPERIMENT
    # ---------------------------------------------------
    def run_experiment(self, model_name, k_fold=None, imputer=None, outlier_handler=None):

        wrapper = self.registry.get_model(model_name)

        if imputer or outlier_handler:
            preprocessor = Preprocessor(
                self.X,
                imputer=imputer or self.imputer,
                outlier_handler=outlier_handler or self.outlier_handler
            ).build()
        else:
            preprocessor = self.preprocessor

        wrapper.build_pipeline(preprocessor)
        pipeline = wrapper.get_pipeline()

        # ✅ K-FOLD mode
        if k_fold:
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

            y_pred = cross_val_predict(
                pipeline,   # ✅ safe now
                self.X_train,
                self.y_train,
                cv=kf,
                n_jobs=8
            )

            result = {
                "model": model_name,
                "mode": "k-fold",
                "type": "baseline",
                "k": k_fold,
                "R2": r2_score(self.y_train, y_pred),
                "MSE": mean_squared_error(self.y_train, y_pred)
            }

        else:
            # ✅ delegate to runner (clean separation)
            result = self.runner.run(
                model_name,
                wrapper,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            )

            result.update({
                "mode": "train-test",
                "type": "baseline",
                "R2": result["R2"],
                "MSE": result["MSE"]
            })

        self.experiment_results.append(result)
        return result

    # ---------------------------------------------------
    # RUN ALL MODELS
    # ---------------------------------------------------
    def run_all_models(self, k_fold=None):

        results = []

        for model_name in self.registry.get_all_models().keys():
            results.append(self.run_experiment(model_name, k_fold=k_fold))

        return pd.DataFrame(results)

    # ---------------------------------------------------
    # RUN MULTIPLE EXPERIMENTS
    # ---------------------------------------------------
    def run_experiments(self, configs):

        results = []

        for config in configs:
            result = self.run_experiment(**config)
            results.append(result)

        return pd.DataFrame(results)

    # ---------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------
    def grid_search_cv(self, model_name, param_grid, cv=5, scoring="r2"):

        wrapper = self.registry.get_model(model_name)
        wrapper.build_pipeline(self.preprocessor)

        pipeline = wrapper.get_pipeline()   # ✅ FIX

        from sklearn.model_selection import GridSearchCV

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=8
        )

        grid.fit(self.X_train, self.y_train)

        y_pred = grid.best_estimator_.predict(self.X_test)

        result = {
            "model": model_name,
            "type": "tuned",
            "mode": "gridsearch",
            "best_params": grid.best_params_,
            "best_score_cv": grid.best_score_,
            "R2": r2_score(self.y_test, y_pred),
            "MSE": mean_squared_error(self.y_test, y_pred)
        }

        self.experiment_results.append(result)
        return result

    # ---------------------------------------------------
    # TUNING VIA TUNING ENGINE
    # ---------------------------------------------------
    def tune_model(self, model_name, param_grid, search_type="grid", cv=5, n_iter=20):

        wrapper = self.registry.get_model(model_name)
        wrapper.build_pipeline(self.preprocessor)

        pipeline = wrapper.get_pipeline()   # ✅ FIX

        tuner = HyperparameterTuner(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )

        if search_type == "grid":
            result = tuner.grid_search(pipeline, param_grid, cv=cv)
        else:
            result = tuner.random_search(pipeline, param_grid, n_iter=n_iter, cv=cv)

        result.update({
            "model": model_name,
            "type": "tuned",
            "search_type": search_type
        })

        self.experiment_results.append(result)
        return result

    # ---------------------------------------------------
    # RESULTS UTILITIES
    # ---------------------------------------------------
    def get_results_df(self):
        return pd.DataFrame(self.experiment_results)

    def rank_models(self, metric="R2", ascending=False):
        comparator = ModelComparator(self.experiment_results)
        return comparator.rank(metric, ascending)

    def get_best_model(self, metric="R2"):
        comparator = ModelComparator(self.experiment_results)
        return comparator.best_model(metric)

    def compare_models(self):
        comparator = ModelComparator(self.experiment_results)
        return comparator.compare()
