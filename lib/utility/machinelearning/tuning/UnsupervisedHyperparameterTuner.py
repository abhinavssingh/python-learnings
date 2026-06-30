from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, ParameterSampler

from lib.utility.machinelearning.shared.ResultBuilder import ResultBuilder


class UnsupervisedHyperparameterTuner:
    """
    Wrapper-aligned unsupervised tuner.
    """

    def __init__(self, X):
        self.X = X

    def tune(
        self,
        wrapper,
        model_name: str,
        preprocessor,
        search_type: str = "grid",
        param_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Returns run payload dictionaries.

        Each payload contains:
        - result: framework-aligned result row
        - labels: clustering labels (or None)
        - reducer: PCA reducer (or None)
        - pipeline: fitted pipeline (or None)
        """

        if param_config is None:
            if not kwargs:
                raise ValueError("Provide param_config or tuning kwargs")

            param_config = {
                f"model__{k}": v if isinstance(v, list) else [v]
                for k, v in kwargs.items()
                if k not in {"n_iter", "random_state"}
            }

        if search_type == "grid":
            param_sets = list(ParameterGrid(param_config))
        elif search_type == "random":
            param_sets = list(
                ParameterSampler(
                    param_distributions=param_config,
                    n_iter=kwargs.get("n_iter", 20),
                    random_state=kwargs.get("random_state", 42),
                )
            )
        else:
            raise ValueError(f"Unsupported search_type: {search_type}")

        payloads: List[Dict[str, Any]] = []

        for i, params in enumerate(param_sets):
            exp_id = f"{model_name} | {search_type} | run_{i}"

            try:
                reducer = self._build_reducer()

                wrapper.build_pipeline(
                    preprocessor,
                    extra_steps=[("reducer", reducer)]
                )

                pipeline = wrapper.get_pipeline()
                pipeline.set_params(**params)

                if hasattr(pipeline, "fit_predict"):
                    labels = pipeline.fit_predict(self.X)
                else:
                    pipeline.fit(self.X)
                    labels = pipeline.predict(self.X)

                metrics = self._normalize_metrics(wrapper.evaluate(self.X, labels))

                extra = {
                    "search_type": search_type,
                    "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                    "noise_points": int(np.sum(labels == -1)) if -1 in labels else 0,
                }

                for key, value in params.items():
                    extra[f"param_{key}"] = value

                result = ResultBuilder.build(
                    model=model_name,
                    family=getattr(wrapper, "family", "unknown"),
                    experiment=exp_id,
                    task="unsupervised",
                    mode="fit_predict",
                    result_type="tuned",
                    extra=extra,
                    **metrics,
                )

                payload = {
                    "result": result,
                    "labels": labels,
                    "reducer": reducer,
                    "pipeline": pipeline,
                }

            except Exception as e:
                result = ResultBuilder.build(
                    model=model_name,
                    family=getattr(wrapper, "family", "unknown"),
                    experiment=exp_id,
                    task="unsupervised",
                    mode="fit_predict",
                    result_type="failed",
                    extra={
                        "search_type": search_type,
                        "params": params,
                        "error": str(e),
                    },
                )

                payload = {
                    "result": result,
                    "labels": None,
                    "reducer": None,
                    "pipeline": None,
                }

            payloads.append(payload)

        return payloads

    def _normalize_metrics(self, metrics):
        clean_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float)) or value is None:
                clean_metrics[key] = value

        return clean_metrics

    def _build_reducer(self):
        n_features = self.X.shape[1]
        n_components = max(1, min(10, n_features))
        return PCA(n_components=n_components, random_state=42)
