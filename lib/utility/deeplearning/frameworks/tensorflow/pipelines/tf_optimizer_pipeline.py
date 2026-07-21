"""
TensorFlow Optimizer Pipeline

Purpose:
    Compare TensorFlow optimizers.

Examples:
    - adam
    - adamw
    - sgd
    - rmsprop
    - adagrad
"""

import time
from copy import deepcopy

import pandas as pd

from lib.utility.common.experiment_result import ExperimentResult
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)


class TensorFlowOptimizerPipeline:

    def __init__(
        self,
        model_wrapper_class,
        base_config,
        optimizers: list[str],
        model_kwargs: dict,
    ):
        self.model_wrapper_class = model_wrapper_class
        self.base_config = base_config
        self.optimizers = optimizers
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
        Compare multiple optimizers.
        """

        results = []

        for optimizer in self.optimizers:

            print(
                f"\nRunning Optimizer: {optimizer}"
            )

            config = deepcopy(self.base_config)

            config.optimizer = optimizer

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

            duration = round(
                time.time() - start_time,
                2,
            )

            result = ExperimentResult(
                model_name=model_wrapper.model_name,
                model_type=model_wrapper.__class__.__name__,
                training_time=duration,
                metrics=metrics,
                parameters={
                    "optimizer": optimizer,
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
        Convert ExperimentResult list into DataFrame.
        """

        rows = []

        for result in results:

            row = {
                "optimizer": result.parameters.get(
                    "optimizer"
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

        return df.reset_index(drop=True)
