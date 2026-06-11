import pandas as pd


class ClassificationModelComparator:
    """
    Comparator for classification model results.
    Supports ranking, best model selection, and comparison.
    """

    def __init__(self, results):
        """
        results: list of dict OR DataFrame
        """
        self.df = pd.DataFrame(results)

        if self.df.empty:
            raise ValueError("No results available for comparison")

    # ---------------------------------------------------
    # INTERNAL: VALIDATE METRIC
    # ---------------------------------------------------
    def _validate_metric(self, metric):
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        return metric

    # ---------------------------------------------------
    # RANK MODELS
    # ---------------------------------------------------
    def rank(self, metric="accuracy", ascending=False):
        """
        Rank models based on metric.
        """
        metric = self._validate_metric(metric)

        df_sorted = self.df.sort_values(by=metric, ascending=ascending)

        df_sorted["rank"] = range(1, len(df_sorted) + 1)

        return df_sorted

    # ---------------------------------------------------
    # BEST MODEL
    # ---------------------------------------------------
    def best_model(self, metric="accuracy"):
        """
        Get best performing model.
        """
        metric = self._validate_metric(metric)

        best_idx = self.df[metric].idxmax()

        return self.df.loc[best_idx]

    # ---------------------------------------------------
    # COMPARE MODELS
    # ---------------------------------------------------
    def compare(self):
        """
        Returns full DataFrame (clean view).
        """
        return self.df.sort_values(by="accuracy", ascending=False)

    # ---------------------------------------------------
    # BEST PER MODEL (important for tuning)
    # ---------------------------------------------------
    def best_per_model(self, metric="accuracy"):
        """
        Get best configuration per model.
        """
        metric = self._validate_metric(metric)

        idx = self.df.groupby("model")[metric].idxmax()

        return self.df.loc[idx].sort_values(metric, ascending=False)

    # ---------------------------------------------------
    # FILTER BY MODE
    # ---------------------------------------------------
    def filter_mode(self, mode):
        """
        Filter results by mode (train-test, k-fold, tuning)
        """
        return self.df[self.df["mode"] == mode]

    # ---------------------------------------------------
    # FILTER BY TYPE
    # ---------------------------------------------------
    def filter_type(self, type_):
        """
        Filter by baseline / tuned
        """
        return self.df[self.df["type"] == type_]
