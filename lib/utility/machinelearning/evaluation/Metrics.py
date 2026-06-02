from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, root_mean_squared_error


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
    def classification(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted")
        }
