from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from lib.utility.machinelearning.evaluation.Metrics import Metrics
from lib.utility.machinelearning.shared.DataCleaner import DataCleaner
from lib.utility.machinelearning.shared.Formatter import Formatter


class ClassificationHyperparameterTuner:
    """
    Generic classification tuner supporting dynamic kwargs.
    Works with any classification model.
    """

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # ---------------------------------------------------
    # MAIN ENTRY
    # ---------------------------------------------------

    def tune(
        self,
        pipeline,
        model_name: str,
        search_type: str = "grid",
        param_config: Dict[str, Any] = None,
        **kwargs
    ) -> List[Dict]:

        print(f"🔧 Tuning {model_name} using {search_type}")

        # ✅ NEW: auto-build param_config from kwargs if not provided
        if param_config is None:

            if not kwargs:
                raise ValueError("Either param_config or model parameters (**kwargs) must be provided")

            # ✅ prefix parameters automatically
            param_config = {
                f"model__{k}": v if isinstance(v, list) else [v]
                for k, v in kwargs.items()
            }

        # ✅ routing
        if search_type == "grid":
            return self._grid_search(pipeline, model_name, param_config, **kwargs)

        elif search_type == "random":
            return self._random_search(pipeline, model_name, param_config, **kwargs)

        elif search_type == "none":
            pipeline.fit(self.X_train, self.y_train)
            return [self._evaluate_model(pipeline, model_name)]

        else:
            raise ValueError(f"Unsupported search_type: {search_type}")
    # ---------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------

    def _grid_search(self, pipeline, model_name, param_grid, **kwargs):

        if not param_grid:
            raise ValueError(f"param_config missing for {model_name}")

        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring=kwargs.get("scoring", "accuracy"),
            cv=kwargs.get("cv", 5),
            n_jobs=kwargs.get("n_jobs", 1)  # ✅ avoid nested parallelism
        )

        grid.fit(self.X_train, self.y_train)

        return self._process_results(grid, model_name, "gridsearch", "grid")

    # ---------------------------------------------------
    # RANDOM SEARCH
    # ---------------------------------------------------
    def _random_search(self, pipeline, model_name, param_dist, **kwargs):

        if not param_dist:
            raise ValueError(f"param_config missing for {model_name}")

        search = RandomizedSearchCV(
            pipeline,
            param_dist,
            n_iter=kwargs.get("n_iter", 20),
            scoring=kwargs.get("scoring", "accuracy"),
            cv=kwargs.get("cv", 5),
            n_jobs=kwargs.get("n_jobs", 1),  # ✅ safe
            random_state=42
        )

        search.fit(self.X_train, self.y_train)

        return self._process_results(search, model_name, "random_search", "random")

    # ---------------------------------------------------
    # PROCESS RESULTS
    # ---------------------------------------------------
    def _process_results(self, search_obj, model_name, mode, search_type):

        cleaner = DataCleaner(pd.DataFrame())

        rows = cleaner.flatten_cv_results(
            search_obj.cv_results_,
            model_name=model_name,
            mode=mode
        )

        exp_name = Formatter.build(
            model_name=model_name,
            mode=mode,
            search_type=search_type
        )

        for i, row in enumerate(rows):
            row.update({
                "experiment": exp_name,
                "type": "tuned",
                "search_type": search_type,
                "iteration": i
            })

        # ✅ Add best result
        best_result = self._build_best_result(
            search_obj, model_name, mode, search_type
        )
        rows.append(best_result)

        # ✅ sort results
        rows = sorted(
            rows,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        return rows

    # ---------------------------------------------------
    # BEST RESULT
    # ---------------------------------------------------
    def _build_best_result(self, search_obj, model_name, mode, search_type):

        exp_name = Formatter.build(
            model_name=model_name,
            mode=f"{mode}_best",
            search_type=search_type
        )

        result = {
            "model": model_name,
            "experiment": exp_name,
            "mode": f"{mode}_best",
            "type": "tuned",
            "search_type": search_type,
            "best_score_cv": search_obj.best_score_
        }

        # ✅ flatten params
        for k, v in search_obj.best_params_.items():
            result[f"param_{k}"] = v

        # ✅ evaluate on test data
        if self.X_test is not None and self.y_test is not None:

            best_model = search_obj.best_estimator_

            y_pred = best_model.predict(self.X_test)

            y_proba = None
            if hasattr(best_model, "predict_proba"):
                try:
                    y_proba = best_model.predict_proba(self.X_test)
                except Exception:
                    y_proba = None

            result.update(
                Metrics.classification(
                    self.y_test,
                    y_pred,
                    y_proba=y_proba
                )
            )

        return result

    # ---------------------------------------------------
    # NO-TUNING CASE
    # ---------------------------------------------------
    def _evaluate_model(self, pipeline, model_name):

        y_pred = pipeline.predict(self.X_test)

        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(self.X_test)
            except Exception:
                y_proba = None

        metrics = Metrics.classification(
            self.y_test,
            y_pred,
            y_proba=y_proba
        )

        return {
            "model": model_name,
            "experiment": f"{model_name} | no-tuning",
            "mode": "train",
            "type": "baseline",
            **metrics
        }
