from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HyperparameterTuner:
    """
    Utility class for hyperparameter tuning using GridSearchCV.
    Decoupled from model training to maintain separation of concerns.
    """

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        """
        Initialize the tuner with training data.

        Parameters:
        - X_train: Training features
        - y_train: Training target
        - X_test: Test features (optional, for evaluation)
        - y_test: Test target (optional, for evaluation)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def grid_search(self, pipeline, param_grid, cv=5, scoring='r2', n_jobs=8):
        """
        Perform GridSearchCV on a pipeline.

        Parameters:
        - pipeline: sklearn Pipeline object
        - param_grid: Dictionary of parameters to search
        - cv: Number of cross-validation folds (default: 5)
        - scoring: Scoring metric (default: 'r2')
        - n_jobs: Number of parallel jobs (default: -1 for all cores)

        Returns:
        Dictionary with grid search results.

        Raises:
        ValueError if X_train or y_train is None
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided to HyperparameterTuner")

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )

        grid.fit(self.X_train, self.y_train)

        results = {
            "mode": "gridsearch",
            "best_params": grid.best_params_,
            "best_score_cv": grid.best_score_,
            "cv_results": grid.cv_results_
        }

        # ✅ keep full params (optional but useful)
        results["best_params"] = results["best_params"]

        # ✅ ✅ CRITICAL FIX: flatten params
        for k, v in results["best_params"].items():
            results[f"param_{k}"] = v

        if self.X_test is not None and self.y_test is not None:
            y_pred = grid.best_estimator_.predict(self.X_test)
            results["test_metrics"] = {
                "MSE": mean_squared_error(self.y_test, y_pred),
                "R2": r2_score(self.y_test, y_pred)
            }

        return results

    def random_search(self, pipeline, param_distributions, cv=5, n_iter=20, scoring='r2', n_jobs=8):
        """
        Perform RandomizedSearchCV on a pipeline.

        Parameters:
        - pipeline: sklearn Pipeline object
        - param_distributions: Dictionary of parameter distributions
        - cv: Number of cross-validation folds (default: 5)
        - n_iter: Number of iterations (default: 20)
        - scoring: Scoring metric (default: 'r2')
        - n_jobs: Number of parallel jobs (default: -1 for all cores)

        Returns:
        Dictionary with random search results.

        Raises:
        ValueError if X_train or y_train is None
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided to HyperparameterTuner")

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

        results = {
            "mode": "random_search",
            "best_params": search.best_params_,
            "best_score_cv": search.best_score_,
            "cv_results": search.cv_results_
        }

        # ✅ ✅ CRITICAL FIX: flatten params
        for k, v in results["best_params"].items():
            results[f"param_{k}"] = v

        if self.X_test is not None and self.y_test is not None:
            y_pred = search.best_estimator_.predict(self.X_test)
            results["test_metrics"] = {
                "MSE": mean_squared_error(self.y_test, y_pred),
                "R2": r2_score(self.y_test, y_pred)
            }

        return results
