from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target

AverageType = Literal["micro", "macro", "samples", "weighted", "binary"]


class Metrics:
    """
    Central metrics utility.
    Supports both regression and classification.
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

        # ---------------------------------------------------
        # ROC-AUC
        # ---------------------------------------------------
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

        # ---------------------------------------------------
        # LOG LOSS
        # ---------------------------------------------------
        if y_proba is not None:
            try:
                result["log_loss"] = log_loss(y_true, y_proba)
            except Exception:
                result["log_loss"] = None

        # ---------------------------------------------------
        # CONFUSION MATRIX
        # ---------------------------------------------------
        if include_confusion_matrix:
            try:
                result["confusion_matrix"] = confusion_matrix(y_true, y_pred)
            except Exception:
                result["confusion_matrix"] = None

        # ---------------------------------------------------
        # CLASSIFICATION REPORT
        # ---------------------------------------------------
        if include_report:
            result["classification_report"] = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )

        # ---------------------------------------------------
        # CURVES ✅ FIXED
        # ---------------------------------------------------
        if include_curves and y_proba is not None:

            try:

                # ✅ BINARY FIX
                if target_type == "binary":

                    # ✅ convert to numeric (CRITICAL FIX)
                    classes = np.unique(y_true)

                    # map labels → 0 / 1
                    label_map = {classes[0]: 0, classes[1]: 1}
                    y_true_bin = np.vectorize(label_map.get)(y_true)

                    # ✅ safe probability extraction
                    y_score = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba

                    # ✅ ROC now works
                    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)

                    precision_c, recall_c, _ = precision_recall_curve(y_true_bin, y_score)

                    result["roc_curve"] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds   # ✅ CRITICAL
                    }

                    result["pr_curve"] = {"precision": precision_c, "recall": recall_c}

                # ✅ MULTICLASS
                elif target_type == "multiclass":

                    classes = np.unique(y_true)
                    y_true_bin = label_binarize(y_true, classes=classes)

                    fpr, tpr = {}, {}

                    for i in range(len(classes)):
                        fpr[i], tpr[i], _ = roc_curve(
                            y_true_bin[:, i],
                            y_proba[:, i]
                        )

                    result["roc_curve"] = {"multi": True, "fpr": fpr, "tpr": tpr}

                # ✅ MULTILABEL
                elif target_type == "multilabel-indicator":

                    n_labels = y_proba.shape[1]
                    fpr, tpr = {}, {}

                    for i in range(n_labels):
                        fpr[i], tpr[i], _ = roc_curve(
                            y_true[:, i],
                            y_proba[:, i]
                        )

                    result["roc_curve"] = {"multilabel": True, "fpr": fpr, "tpr": tpr}

            except Exception as e:
                print("⚠️ ROC generation failed:", str(e))  # ✅ DEBUG instead of silent fail

        return result
