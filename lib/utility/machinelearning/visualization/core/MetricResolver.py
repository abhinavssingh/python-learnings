import pandas as pd


class MetricResolver:
    """
    Detects and manages metrics dynamically.
    Works for regression, classification, etc.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_numeric_metrics(self):
        """Return all numeric columns (potential metrics)"""
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        excluded = ["iteration"]
        return [col for col in numeric_cols if col not in excluded]

    def detect_problem_type(self):
        """Infer ML type based on metrics"""
        cols = self.get_numeric_metrics()

        if "R2" in cols or "MSE" in cols:
            return "regression"

        if "accuracy" in cols or "f1" in cols:
            return "classification"

        return "unknown"

    def get_default_metrics(self):
        """Return best guess metrics for plotting"""
        problem = self.detect_problem_type()

        if problem == "regression":
            return ("MSE", "R2")

        if problem == "classification":
            return ("accuracy", "f1")

        # fallback
        metrics = self.get_numeric_metrics()
        return metrics[:2] if len(metrics) >= 2 else (None, None)
