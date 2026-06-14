import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HyperparameterTuner:
    """
    Framework-aligned tuner supporting classification + regression.
    Wrapper-driven, artifact-aware, AutoML-ready.
    """

    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # ---------------------------------------------------
    # INTERNAL: CLEAN CV RESULTS
    # ---------------------------------------------------
    def _process_cv_results(self, cv_results, model_name, mode):

        from lib.utility.machinelearning.shared.DataCleaner import DataCleaner

        cleaner = DataCleaner(pd.DataFrame())

        return cleaner.flatten_cv_results(
            cv_results,
            model_name=model_name,
            mode=mode
        )

    # ---------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------

    def grid_search(self, wrapper, model_name, param_grid,
                    cv=5, scoring=None, n_jobs=8):

        try:
            wrapper.build_pipeline()
            pipeline = wrapper.get_pipeline()

            # ✅ FIX: enforce regression scoring
            scoring = scoring or "neg_mean_squared_error"

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs
            )

            grid.fit(self.X_train, self.y_train)

            results = self._process_cv_results(
                grid.cv_results_,
                model_name=model_name,
                mode="gridsearch"
            )

            best_result = self._build_best_result(
                wrapper,
                grid,
                model_name,
                mode="gridsearch"
            )

            results.append(best_result)

            return results

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

    def random_search(self, wrapper, model_name, param_dist,
                      cv=5, n_iter=20, scoring=None, n_jobs=8):

        try:
            wrapper.build_pipeline()
            pipeline = wrapper.get_pipeline()

            # ✅ FIX: enforce regression scoring
            scoring = scoring or "neg_mean_squared_error"

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                cv=cv,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=42
            )

            search.fit(self.X_train, self.y_train)

            results = self._process_cv_results(
                search.cv_results_,
                model_name=model_name,
                mode="random_search"
            )

            best_result = self._build_best_result(
                wrapper,
                search,
                model_name,
                mode="random_search"
            )

            results.append(best_result)

            return results

        except Exception as e:
            return [{
                "model": model_name,
                "type": "failed",
                "experiment": f"{model_name} | random_search",
                "error": str(e)
            }]

    # ---------------------------------------------------
    # BEST RESULT (FRAMEWORK-ALIGNED)
    # ---------------------------------------------------
    def _build_best_result(self, wrapper, search_obj, model_name, mode):

        result = {
            "model": model_name,
            "experiment": f"{model_name} | {mode}_best",
            "mode": f"{mode}_best",
            "type": "tuned",
            "best_score_cv": search_obj.best_score_,
            "task": getattr(wrapper, "task", "unknown"),
            "family": getattr(wrapper, "family", "unknown")
        }

        # ✅ flatten params
        for k, v in search_obj.best_params_.items():
            result[f"param_{k}"] = v

        # ✅ evaluation using wrapper (IMPORTANT)
        if self.X_test is not None and self.y_test is not None:

            try:
                best_model = search_obj.best_estimator_

                y_pred = best_model.predict(self.X_test)

                try:
                    y_proba = best_model.predict_proba(self.X_test)
                except Exception:
                    y_proba = None

                metrics = wrapper.evaluate(self.y_test, y_pred, y_proba)

                artifacts, metrics = self._extract_artifacts(metrics)

                result.update(metrics)
                result["artifacts"] = artifacts

            except Exception as e:
                result["error"] = str(e)

        return result

    # ---------------------------------------------------
    # ARTIFACT SPLITTING
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
