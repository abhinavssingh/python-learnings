"""
TensorFlow Model Pipeline

Purpose:
    - Train TensorFlow model
    - Evaluate model
    - Generate predictions
    - Produce experiment results

This serves as the baseline pipeline for all
future deep learning experiments.
"""

import time

import pandas as pd

from lib.utility.common.experiment_result import ExperimentResult
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)


class TensorFlowModelPipeline:
    def __init__(
        self,
        model_wrapper,
        config,
    ):
        self.model_wrapper = model_wrapper
        self.config = config

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        validation_data=None,
    ) -> ExperimentResult:
        """
        Execute complete training pipeline.
        """

        utility = TensorFlowModelUtility(
            model_wrapper=self.model_wrapper,
            config=self.config,
        )

        start_time = time.time()

        # Compile model
        utility.compile()

        # Train model
        history = utility.train(
            X_train=X_train,
            y_train=y_train,
            validation_data=validation_data,
        )

        # Evaluate model
        metrics = utility.evaluate(
            X_test=X_test,
            y_test=y_test,
        )

        # Predictions
        predictions = utility.predict(X_test)

        duration = round(
            time.time() - start_time,
            2,
        )

        result = ExperimentResult(
            model_name=self.model_wrapper.model_name,
            model_type=self.model_wrapper.__class__.__name__,
            training_time=duration,
            metrics=metrics,
            history=history,
            artifacts={
                "predictions": predictions,
            },
        )

        return result

    @staticmethod
    def results_to_dataframe(
        results: list[ExperimentResult],
    ) -> pd.DataFrame:
        """
        Convert experiment results into dataframe.
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

        return pd.DataFrame(rows)
