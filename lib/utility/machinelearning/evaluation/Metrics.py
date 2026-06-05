from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
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
    def classification(y_true, y_pred, include_confusion_matrix=False):

        result = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # ✅ optional matrix
        if include_confusion_matrix:
            result["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        return result
