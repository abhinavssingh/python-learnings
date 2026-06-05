import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from lib.utility.machinelearning.shared.DataCleaner import DataCleaner


class HyperparameterTuner:
    """
    Utility class for hyperparameter tuning.
    Produces CLEAN and PLOTTABLE outputs using DataCleaner.
    """

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # ---------------------------------------------------
    # INTERNAL: CLEAN CV RESULTS USING DataCleaner
    # ---------------------------------------------------
    def _process_cv_results(self, cv_results, model_name, mode):
        """
        Uses DataCleaner to convert cv_results_ into clean rows
        """

        cleaner = DataCleaner(pd.DataFrame())

        rows = cleaner.flatten_cv_results(
            cv_results,
            model_name=model_name,
            mode=mode
        )

        return rows

    # ---------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------
    def grid_search(self, pipeline, model_name, param_grid,
                    cv=5, scoring='r2', n_jobs=8):

        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided")

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )

        grid.fit(self.X_train, self.y_train)

        # ✅ ✅ Convert full grid results → rows
        results = self._process_cv_results(
            grid.cv_results_,
            model_name=model_name,
            mode="gridsearch"
        )

        # ✅ ✅ Add best result separately
        best_result = {
            "model": model_name,
            "type": "tuned",
            "mode": "gridsearch_best",
            "best_score_cv": grid.best_score_,
        }

        # ✅ flatten best params
        for k, v in grid.best_params_.items():
            best_result[f"param_{k}"] = v

        # ✅ add test metrics if provided
        if self.X_test is not None and self.y_test is not None:
            y_pred = grid.best_estimator_.predict(self.X_test)
            best_result["test_metrics"] = {
                "MSE": mean_squared_error(self.y_test, y_pred),
                "R2": r2_score(self.y_test, y_pred)
            }

        results.append(best_result)

        return results

    # ---------------------------------------------------
    # RANDOM SEARCH
    # ---------------------------------------------------
    def random_search(self, pipeline, model_name, param_distributions,
                      cv=5, n_iter=20, scoring='r2', n_jobs=8):

        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided")

        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            cv=cv,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42
        )

        search.fit(self.X_train, self.y_train)

        # ✅ ✅ Convert full search results → rows
        results = self._process_cv_results(
            search.cv_results_,
            model_name=model_name,
            mode="random_search"
        )

        # ✅ ✅ Best result
        best_result = {
            "model": model_name,
            "type": "tuned",
            "mode": "random_search_best",
            "best_score_cv": search.best_score_,
        }

        # ✅ flatten params
        for k, v in search.best_params_.items():
            best_result[f"param_{k}"] = v

        # ✅ test metrics
        if self.X_test is not None and self.y_test is not None:
            y_pred = search.best_estimator_.predict(self.X_test)
            best_result["test_metrics"] = {
                "MSE": mean_squared_error(self.y_test, y_pred),
                "R2": r2_score(self.y_test, y_pred)
            }

        results.append(best_result)

        return results
