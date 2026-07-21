"""
TensorFlow Batch Size Pipeline

Purpose:
    Compare different batch sizes under
    identical training conditions.

Examples:
    - 16
    - 32
    - 64
    - 128
    - 256
"""

import time
from copy import deepcopy

import pandas as pd

from lib.utility.common.experiment_result import ExperimentResult
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)


class TensorFlowBatchSizePipeline:

    def __init__(
        self,
        model_wrapper_class,
        base_config,
        batch_sizes: list[int],
        model_kwargs: dict,
    ):
        self.model_wrapper_class = model_wrapper_class
        self.base_config = base_config
        self.batch_sizes = batch_sizes
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
        Compare different batch sizes.
        """

        results = []

        for batch_size in self.batch_sizes:

            print(
                f"\nRunning Batch Size: {batch_size}"
            )

            config = deepcopy(self.base_config)

            config.batch_size = batch_size

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
                    "batch_size": batch_size,
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
        Convert results into comparison dataframe.
        """

        rows = []

        for result in results:

            row = {
                "batch_size": result.parameters.get(
                    "batch_size"
                ),
                "model_name": result.model_name,
                "model_type": result.model_type,
                "training_time": result.training_time,
            }

            row.update(result.metrics)

            rows.append(row)

        df = pd.DataFrame(rows)

        sort_column = None

        if "accuracy" in df.columns:
            sort_column = "accuracy"
        elif "val_accuracy" in df.columns:
            sort_column = "val_accuracy"
        elif "loss" in df.columns:
            sort_column = "loss"

        if sort_column:

            ascending = sort_column == "loss"

            df = df.sort_values(
                by=sort_column,
                ascending=ascending,
            )

        return df.reset_index(drop=True)
