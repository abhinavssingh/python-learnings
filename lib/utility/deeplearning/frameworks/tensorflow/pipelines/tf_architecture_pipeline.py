"""
TensorFlow Architecture Pipeline

Purpose:
    Compare multiple TensorFlow architectures
    under identical training conditions.

Supported Examples:
    - MLP
    - ParallelMLP
    - MixedMLP
    - CNN
    - ResNet
    - LSTM
    - Transformer
"""

import time
from typing import List

import pandas as pd

from lib.utility.common.experiment_result import ExperimentResult
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)


class TensorFlowArchitecturePipeline:
    """
    Architecture comparison pipeline.
    """

    def __init__(
        self,
        model_wrappers: List,
        config,
    ):
        self.model_wrappers = model_wrappers
        self.config = config

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        validation_data=None,
    ) -> pd.DataFrame:
        """
        Execute architecture comparison.
        """

        results = []

        for model_wrapper in self.model_wrappers:

            print(
                f"\nTraining Model: "
                f"{model_wrapper.model_name}"
            )

            utility = TensorFlowModelUtility(
                model_wrapper=model_wrapper,
                config=self.config,
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
                history=history,
            )

            results.append(result)

        return self.results_to_dataframe(results)

    @staticmethod
    def results_to_dataframe(
        results: List[ExperimentResult],
    ) -> pd.DataFrame:
        """
        Convert ExperimentResult objects
        to comparison dataframe.
        """

        rows = []

        for result in results:

            row = {
                "model_name": result.model_name,
                "model_type": result.model_type,
                "training_time": result.training_time,
            }

            row.update(result.metrics)

            rows.append(row)

        df = pd.DataFrame(rows)

        metric_columns = [
            col
            for col in df.columns
            if col not in {
                "model_name",
                "model_type",
                "training_time",
            }
        ]

        if "accuracy" in metric_columns:
            df = df.sort_values(
                by="accuracy",
                ascending=False,
            )

        return df.reset_index(drop=True)
