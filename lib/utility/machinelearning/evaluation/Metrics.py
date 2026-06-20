from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target

AverageType = Literal["micro", "macro", "samples", "weighted", "binary"]


class Metrics:
    """
    Pure computation layer.
    No formatting, no UI logic.
    """

    @staticmethod
    def regression(y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred)
        }

    @staticmethod
    def classification(
        y_true,
        y_pred,
        y_proba=None,
        average="weighted",
        include_curves=False,
        include_report=False,
        include_confusion_matrix=False
    ):

        result = {}

        target_type = type_of_target(y_true)

        # ✅ averaging
        if target_type == "binary":
            avg: AverageType = "weighted"
        elif target_type == "multiclass":
            avg = average
        elif target_type == "multilabel-indicator":
            avg = "samples"
        else:
            avg = average

        # ✅ basic metrics
        result.update({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
            "problem_type": target_type
        })

        # ✅ ROC-AUC
        if y_proba is not None:
            try:
                if target_type == "binary":
                    y_score = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
                    result["roc_auc"] = roc_auc_score(y_true, y_score)

                elif target_type == "multiclass":
                    result["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")

                elif target_type == "multilabel-indicator":
                    result["roc_auc"] = roc_auc_score(y_true, y_proba, average="samples")

            except Exception:
                result["roc_auc"] = None

        # ✅ log loss
        if y_proba is not None:
            try:
                result["log_loss"] = log_loss(y_true, y_proba)
            except Exception:
                result["log_loss"] = None

        # ✅ confusion matrix (raw)
        if include_confusion_matrix:
            try:
                result["confusion_matrix"] = confusion_matrix(y_true, y_pred)
            except Exception:
                result["confusion_matrix"] = None

        # ✅ classification report
        if include_report:
            result["classification_report"] = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )

        # ✅ curves (RAW ONLY)
        if include_curves and y_proba is not None:

            try:

                if target_type == "binary":

                    classes = np.unique(y_true)
                    label_map = {classes[0]: 0, classes[1]: 1}
                    y_true_bin = np.vectorize(label_map.get)(y_true)

                    y_score = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba

                    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
                    precision_c, recall_c, _ = precision_recall_curve(y_true_bin, y_score)

                    result["roc_curve"] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds
                    }

                    result["pr_curve"] = {
                        "precision": precision_c,
                        "recall": recall_c
                    }

                elif target_type == "multiclass":

                    classes = np.unique(y_true)
                    y_true_bin = label_binarize(y_true, classes=classes)

                    fpr, tpr = {}, {}

                    for i in range(len(classes)):
                        fpr[i], tpr[i], _ = roc_curve(
                            y_true_bin[:, i],
                            y_proba[:, i]
                        )

                    result["roc_curve"] = {"fpr": fpr, "tpr": tpr}

                elif target_type == "multilabel-indicator":

                    y_proba_arr = np.array(y_proba)

                    if isinstance(y_proba, list):
                        y_proba_arr = np.column_stack([
                            p[:, 1] if p.ndim > 1 else p
                            for p in y_proba
                        ])

                    fpr, tpr = {}, {}

                    for i in range(y_proba_arr.shape[1]):

                        # ✅ skip invalid labels
                        if len(np.unique(y_true.iloc[:, i])) < 2:
                            continue

                        fpr[i], tpr[i], _ = roc_curve(
                            y_true.iloc[:, i],
                            y_proba_arr[:, i]
                        )

                    result["roc_curve"] = {"fpr": fpr, "tpr": tpr}

            except Exception:
                pass

        return result

    @staticmethod
    def unsupervised(X, labels):

        unique_labels = set(labels)

        # ✅ Edge case: single cluster
        if len(unique_labels) <= 1:
            return {
                "silhouette_score": None,
                "davies_bouldin": None,
                "calinski_harabasz": None
            }

        return {
            "silhouette_score": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels)
        }
