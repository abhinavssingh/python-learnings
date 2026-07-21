"""
TensorFlow Regularization Pipeline

Purpose:
    Compare regularization strategies under
    identical training conditions.

Examples:
    - None
    - l1
    - l2
    - l1_l2

Measures:
    - Accuracy
    - Loss
    - Training Time
"""

import time
from copy import deepcopy

import pandas as pd

from lib.utility.common.experiment_result import ExperimentResult
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)


class TensorFlowRegularizationPipeline:

    def __init__(
        self,
        model_wrapper_class,
        base_config,
        regularizations: list[str | None],
        model_kwargs: dict,
    ):
        self.model_wrapper_class = model_wrapper_class
        self.base_config = base_config
        self.regularizations = regularizations
        self.model_kwargs = model_kwargs

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        validation_data=None,
    ) -> pd.DataFrame:
        """
        Compare different regularization strategies.
        """

        results = []

        for regularization in self.regularizations:

            print(
                f"\nRunning Regularization: "
                f"{regularization}"
            )

            config = deepcopy(self.base_config)

            config.regularizer = regularization

            model_wrapper = self.model_wrapper_class(
                **self.model_kwargs
            )

            utility = TensorFlowModelUtility(
                model_wrapper=model_wrapper,
                config=config,
            )

            start_time = time.time()

            utility.compile()

            history = utility.train(
                X_train=X_train,
                y_train=y_train,
                validation_data=validation_data,
            )

            metrics = utility.evaluate(
                X_test=X_test,
                y_test=y_test,
            )

            training_time = round(
                time.time() - start_time,
                2,
            )

            result = ExperimentResult(
                model_name=model_wrapper.model_name,
                model_type=model_wrapper.__class__.__name__,
                training_time=training_time,
                metrics=metrics,
                parameters={
                    "regularization": (
                        regularization
                        if regularization is not None
                        else "none"
                    )
                },
                history=history,
            )

            results.append(result)

        return self.results_to_dataframe(results)

    @staticmethod
    def results_to_dataframe(
        results: list[ExperimentResult],
    ) -> pd.DataFrame:
        """
        Convert ExperimentResult objects
        into comparison dataframe.
        """

        rows = []

        for result in results:

            row = {
                "regularization": result.parameters.get(
                    "regularization"
                ),
                "model_name": result.model_name,
                "model_type": result.model_type,
                "training_time": result.training_time,
            }

            row.update(result.metrics)

            rows.append(row)

        df = pd.DataFrame(rows)

        if "accuracy" in df.columns:

            df = df.sort_values(
                by="accuracy",
                ascending=False,
            )

        elif "loss" in df.columns:

            df = df.sort_values(
                by="loss",
                ascending=True,
            )

        return df.reset_index(drop=True)
