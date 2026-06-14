from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from lib.utility.machinelearning.shared.DataCleaner import DataCleaner
from lib.utility.machinelearning.shared.Formatter import Formatter


class ClassificationHyperparameterTuner:
    """
    Wrapper-aligned classification tuner.
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
        wrapper,                         # ✅ changed (pass wrapper, not pipeline)
        model_name: str,
        search_type: str = "grid",
        param_config: Dict[str, Any] = None,
        **kwargs
    ) -> List[Dict]:

        print(f"🔧 Tuning {model_name} using {search_type}")

        if param_config is None:
            if not kwargs:
                raise ValueError("Provide param_config or tuning kwargs")

            param_config = {
                f"model__{k}": v if isinstance(v, list) else [v]
                for k, v in kwargs.items()
            }

        pipeline = wrapper.get_pipeline()

        if search_type == "grid":
            return self._grid_search(wrapper, pipeline, model_name, param_config, **kwargs)

        elif search_type == "random":
            return self._random_search(wrapper, pipeline, model_name, param_config, **kwargs)

        elif search_type == "none":
            pipeline.fit(self.X_train, self.y_train)
            return [self._evaluate(wrapper, pipeline, model_name)]

        else:
            raise ValueError(f"Unsupported search_type: {search_type}")

    # ---------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------
    def _grid_search(self, wrapper, pipeline, model_name, param_grid, **kwargs):

        try:
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=kwargs.get("scoring", "accuracy"),
                cv=kwargs.get("cv", 5),
                n_jobs=kwargs.get("n_jobs", -1)
            )

            grid.fit(self.X_train, self.y_train)

            return self._process_results(wrapper, grid, model_name, "gridsearch", "grid")

        except Exception as e:
            return [{
                "model": model_name,
                "type": "failed",
                "experiment": f"{model_name} | gridsearch",
                "error": str(e)
            }]

    # ---------------------------------------------------
    # RANDOM SEARCH
    # ---------------------------------------------------
    def _random_search(self, wrapper, pipeline, model_name, param_dist, **kwargs):

        try:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=kwargs.get("n_iter", 20),
                scoring=kwargs.get("scoring", "accuracy"),
                cv=kwargs.get("cv", 5),
                n_jobs=kwargs.get("n_jobs", -1),
                random_state=42
            )

            search.fit(self.X_train, self.y_train)

            return self._process_results(wrapper, search, model_name, "random_search", "random")

        except Exception as e:
            return [{
                "model": model_name,
                "type": "failed",
                "experiment": f"{model_name} | random",
                "error": str(e)
            }]

    # ---------------------------------------------------
    # PROCESS RESULTS
    # ---------------------------------------------------
    def _process_results(self, wrapper, search_obj, model_name, mode, search_type):

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

        # ✅ enrich rows
        for i, row in enumerate(rows):
            row.update({
                "experiment": exp_name,
                "type": "tuned",
                "search_type": search_type,
                "iteration": i,
                "task": getattr(wrapper, "task", "classification"),
                "family": getattr(wrapper, "family", "unknown")
            })

        # ✅ add best result
        best_result = self._build_best_result(
            wrapper, search_obj, model_name, mode, search_type
        )

        rows.append(best_result)

        # ✅ sort
        rows = sorted(
            rows,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        return rows

    # ---------------------------------------------------
    # BEST RESULT
    # ---------------------------------------------------
    def _build_best_result(self, wrapper, search_obj, model_name, mode, search_type):

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
            "best_score_cv": search_obj.best_score_,
            "task": getattr(wrapper, "task"),
            "family": getattr(wrapper, "family")
        }

        # ✅ params
        for k, v in search_obj.best_params_.items():
            result[f"param_{k}"] = v

        # ✅ evaluate best model
        if self.X_test is not None and self.y_test is not None:

            try:
                best_model = search_obj.best_estimator_

                y_pred = best_model.predict(self.X_test)

                try:
                    y_proba = best_model.predict_proba(self.X_test)
                except Exception:
                    y_proba = None

                metrics = wrapper.evaluate(self.y_test, y_pred, y_proba)

                # ✅ split artifacts
                artifacts, metrics = self._extract_artifacts(metrics)

                result.update(metrics)
                result["artifacts"] = artifacts

            except Exception as e:
                result["error"] = str(e)

        return result

    # ---------------------------------------------------
    # NO-TUNING
    # ---------------------------------------------------
    def _evaluate(self, wrapper, pipeline, model_name):

        y_pred = pipeline.predict(self.X_test)

        try:
            y_proba = pipeline.predict_proba(self.X_test)
        except Exception:
            y_proba = None

        metrics = wrapper.evaluate(self.y_test, y_pred, y_proba)

        artifacts, metrics = self._extract_artifacts(metrics)

        return {
            "model": model_name,
            "experiment": f"{model_name} | no-tuning",
            "mode": "train",
            "type": "baseline",
            "task": getattr(wrapper, "task"),
            "family": getattr(wrapper, "family"),
            **metrics,
            "artifacts": artifacts
        }

    # ---------------------------------------------------
    # ARTIFACT SPLITTER
    # ---------------------------------------------------
    def _extract_artifacts(self, metrics):

        artifact_keys = {"roc_curve", "pr_curve", "confusion_matrix"}

        artifacts = {}
        numeric_metrics = {}

        for k, v in metrics.items():
            if k in artifact_keys:
                artifacts[k] = v
            else:
                numeric_metrics[k] = v

        return artifacts, numeric_metrics
