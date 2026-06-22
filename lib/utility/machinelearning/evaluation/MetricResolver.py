class MetricResolver:

    DEFAULT_METRICS = {
        "classification": ["f1_weighted", "roc_auc"],
        "regression": ["R2", "RMSE"],
        "unsupervised": ["silhouette_score", "calinski_harabasz"]
    }

    METRIC_DIRECTION = {
        "accuracy": 1,
        "f1_weighted": 1,
        "f1_macro": 1,
        "roc_auc": 1,
        "pr_auc": 1,

        "R2": 1,
        "RMSE": -1,
        "MSE": -1,

        "silhouette_score": 1,
        "davies_bouldin": -1,
        "calinski_harabasz": 1
    }

    @staticmethod
    def get_default_metrics(task: str):
        return MetricResolver.DEFAULT_METRICS.get(task, [])

    @staticmethod
    def get_best_metric(task: str):
        metrics = MetricResolver.get_default_metrics(task)
        return metrics[0] if metrics else None

    @staticmethod
    def get_direction(metric: str):
        return MetricResolver.METRIC_DIRECTION.get(metric, 1)
