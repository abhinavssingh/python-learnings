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
        """
            Comprehensive classification metrics.
            """

        result = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        # ✅ ROC-AUC (only if probabilities available)
        if y_proba is not None:
            try:
                if len(set(y_true)) == 2:
                    result["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    result["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            except Exception:
                result["roc_auc"] = None

        # ✅ Log loss
        if y_proba is not None:
            try:
                result["log_loss"] = log_loss(y_true, y_proba)
            except Exception:
                result["log_loss"] = None

        # ✅ Confusion matrix
        if include_confusion_matrix:
            result["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # ✅ Classification report (tabular)
        if include_report:
            report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            result["classification_report"] = report_dict

        # ✅ Curves (ROC + PR)
        if include_curves and y_proba is not None:
            try:
                if len(set(y_true)) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])

                    result["roc_curve"] = {"fpr": fpr, "tpr": tpr}
                    result["pr_curve"] = {"precision": precision, "recall": recall}
            except Exception:
                pass

        return result
