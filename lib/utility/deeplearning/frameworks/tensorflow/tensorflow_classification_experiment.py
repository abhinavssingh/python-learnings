import io
from typing import Any

import numpy as np

from lib.utility.deeplearning.evaluation.classification_evaluator import (
    ClassificationEvaluator,
)
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)
from lib.utility.deeplearning.visualization.confusion_matrix_plot import (
    ConfusionMatrixPlot,
)
from lib.utility.deeplearning.visualization.roc_curve_plot import ROCurvePlot


class ClassificationExperimentRunner:
    @staticmethod
    def run(
        utility: TensorFlowModelUtility,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray],
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_test_labels: np.ndarray | None = None,
        class_labels: list[str] | None = None,
        positive_label_index: int = 1,
        experiment_name: str = "Experiment",
        compile_metrics: list[str] = ("accuracy",),
    ) -> dict[str, Any]:
        utility.compile(metrics=compile_metrics)

        history = utility.train(
            X_train=X_train,
            y_train=y_train,
            validation_data=validation_data,
        )

        test_metrics = utility.evaluate(X_test=X_test, y_test=y_test)

        probs = utility.predict(X_test)
        if probs.ndim == 1 or probs.shape[-1] == 1:
            prob_scores = probs.ravel()
        else:
            prob_scores = probs[:, positive_label_index]

        if probs.ndim == 1 or probs.shape[-1] == 1:
            y_pred = (prob_scores >= 0.5).astype(int)
        else:
            y_pred = np.argmax(probs, axis=1)

        if y_test_labels is None:
            if y_test.ndim > 1 and y_test.shape[1] > 1:
                raw_y_test = np.argmax(y_test, axis=1)
            else:
                raw_y_test = y_test
        else:
            raw_y_test = y_test_labels

        evaluator = ClassificationEvaluator()
        cls_metrics = evaluator.evaluate(y_true=raw_y_test, y_pred=y_pred)

        if class_labels is None:
            class_labels = ["Class 0", "Class 1"]

        confusion_html = ConfusionMatrixPlot.to_html(
            y_true=raw_y_test,
            y_pred=y_pred,
            class_labels=class_labels,
            title=f"{experiment_name} - Confusion Matrix",
            include_plotlyjs=False,
        )

        roc_html, auc_score = ROCurvePlot.to_html(
            y_true=raw_y_test,
            y_prob=prob_scores,
            title=f"{experiment_name} - ROC Curve",
            include_plotlyjs=False,
        )

        history_df = history.to_dataframe()

        final_train_accuracy = (
            float(history_df["accuracy"].iloc[-1])
            if "accuracy" in history_df.columns and not history_df.empty
            else None
        )
        final_val_accuracy = (
            float(history_df["val_accuracy"].iloc[-1])
            if "val_accuracy" in history_df.columns and not history_df.empty
            else None
        )

        model_summary_stream = io.StringIO()

        def _summary_writer(line: str) -> None:
            model_summary_stream.write(line + "\n")

        utility.get_model().summary(print_fn=_summary_writer)

        return {
            "history_df": history_df,
            "model_summary": model_summary_stream.getvalue(),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "classification_metrics": {k: float(v) for k, v in cls_metrics.items()},
            "final_train_accuracy": final_train_accuracy,
            "final_val_accuracy": final_val_accuracy,
            "auc_score": float(auc_score),
            "confusion_html": confusion_html,
            "roc_html": roc_html,
        }
