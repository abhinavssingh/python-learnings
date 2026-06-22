from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
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

    # ======================================================
    # ✅ REGRESSION
    # ======================================================
    @staticmethod
    def regression(y_true, y_pred):
        return {
            "task": "regression",
            "R2": r2_score(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred)
        }

    # ======================================================
    # ✅ CLASSIFICATION (FULL UPDATED ✅)
    # ======================================================
    @staticmethod
    def classification(
        y_true,
        y_pred,
        y_proba=None,
        include_curves=False,
        include_report=False,
        include_confusion_matrix=False
    ):
        result = {}

        target_type = type_of_target(y_true)

        # ✅ BASIC METRICS (MULTI VARIANTS)
        result.update({
            "accuracy": accuracy_score(y_true, y_pred),

            # ✅ Weighted (default production metric)
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),

            # ✅ Macro (imbalance-friendly)
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),

            # ✅ Micro (global)
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),

            "problem_type": target_type
        })

        # ======================================================
        # ✅ ROC-AUC
        # ======================================================
        if y_proba is not None:
            try:
                if target_type == "binary":
                    y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    result["roc_auc"] = roc_auc_score(y_true, y_score)

                elif target_type == "multiclass":
                    result["roc_auc"] = roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="weighted"
                    )

                elif target_type == "multilabel-indicator":
                    result["roc_auc"] = roc_auc_score(
                        y_true,
                        y_proba,
                        average="samples"
                    )

            except Exception:
                result["roc_auc"] = None

        # ======================================================
        # ✅ LOG LOSS
        # ======================================================
        if y_proba is not None:
            try:
                result["log_loss"] = log_loss(y_true, y_proba)
            except Exception:
                result["log_loss"] = None

        # ======================================================
        # ✅ CONFUSION MATRIX (ARTIFACT)
        # ======================================================
        if include_confusion_matrix:
            try:
                result["confusion_matrix"] = confusion_matrix(y_true, y_pred)
            except Exception:
                result["confusion_matrix"] = None

        # ======================================================
        # ✅ CLASSIFICATION REPORT (ARTIFACT)
        # ======================================================
        if include_report:
            result["classification_report"] = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )

        # ======================================================
        # ✅ CURVES (ROC + PR)
        # ======================================================
        if include_curves and y_proba is not None:

            try:

                if target_type == "binary":

                    y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba

                    fpr, tpr, thresholds = roc_curve(y_true, y_score)
                    precision_c, recall_c, pr_thresholds = precision_recall_curve(y_true, y_score)

                    result["roc_curve"] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": thresholds.tolist()
                    }

                    result["pr_curve"] = {
                        "precision": precision_c.tolist(),
                        "recall": recall_c.tolist(),
                        "thresholds": np.append(pr_thresholds, 1.0).tolist()
                    }

                    # ✅ Optimal threshold (Youden’s J)
                    optimal_idx = np.argmax(tpr - fpr)

                    best_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

                    result["best_threshold"] = float(best_threshold)
                    result["best_tpr"] = float(tpr[optimal_idx])
                    result["best_fpr"] = float(fpr[optimal_idx])

                    # ✅ PR-AUC (useful ranking metric)
                    result["pr_auc"] = auc(recall_c, precision_c)

                elif target_type == "multiclass":

                    classes = np.unique(y_true)
                    y_true_bin = label_binarize(y_true, classes=classes)

                    # ✅ --- ROC ---
                    roc_fpr, roc_tpr = {}, {}

                    # ✅ --- PR ---
                    pr_precision, pr_recall = {}, {}

                    auc_values = []

                    for i in range(len(classes)):

                        # ✅ ROC per class
                        roc_fpr[i], roc_tpr[i], _ = roc_curve(
                            y_true_bin[:, i],
                            y_proba[:, i]
                        )

                        # ✅ PR per class
                        p, r, _ = precision_recall_curve(
                            y_true_bin[:, i],
                            y_proba[:, i]
                        )

                        pr_precision[i] = p
                        pr_recall[i] = r

                        # ✅ PR-AUC per class
                        try:
                            auc_values.append(auc(r, p))
                        except Exception:
                            pass

                    # ✅ STORE ROC
                    result["roc_curve"] = {
                        "fpr": {k: v.tolist() for k, v in roc_fpr.items()},
                        "tpr": {k: v.tolist() for k, v in roc_tpr.items()}
                    }

                    # ✅ STORE PR
                    result["pr_curve"] = {
                        "precision": {k: v.tolist() for k, v in pr_precision.items()},
                        "recall": {k: v.tolist() for k, v in pr_recall.items()}
                    }

                    # ✅ MACRO PR-AUC
                    result["pr_auc"] = float(np.mean(np.array(auc_values))) if len(auc_values) > 0 else None

            except Exception:
                pass

        # ✅ FINAL TAG
        result["task"] = "classification"

        return result

    # ======================================================
    # ✅ UNSUPERVISED
    # ======================================================
    @staticmethod
    def unsupervised(X, labels):

        unique_labels = set(labels)

        if len(unique_labels) <= 1:
            return {
                "task": "unsupervised",
                "silhouette_score": None,
                "davies_bouldin": None,
                "calinski_harabasz": None
            }

        return {
            "task": "unsupervised",
            "silhouette_score": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels)
        }
