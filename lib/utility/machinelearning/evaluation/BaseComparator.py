import pandas as pd


class BaseComparator:

    def __init__(self, results):
        self.df = pd.DataFrame(results)

        if self.df.empty:
            raise ValueError("No results available")

    # ✅ Metric direction mapping
    METRIC_DIRECTION = {
        "accuracy": 1,
        "f1_weighted": 1,
        "f1_macro": 1,
        "f1_micro": 1,
        "roc_auc": 1,
        "pr_auc": 1,

        "R2": 1,
        "RMSE": -1,
        "MSE": -1,

        "silhouette_score": 1,
        "davies_bouldin": -1,
        "calinski_harabasz": 1
    }

    def _get_direction(self, metric):
        return self.METRIC_DIRECTION.get(metric, 1)

    def rank(self, metric, ascending=None):

        if metric not in self.df.columns:
            raise ValueError(f"{metric} not found")

        # ✅ auto direction
        if ascending is None:
            ascending = self._get_direction(metric) == -1

        df = self.df.sort_values(metric, ascending=ascending)
        df["rank"] = range(1, len(df) + 1)

        return df

    def best_model(self, metric):
        return self.rank(metric).iloc[0].to_dict()

    def compare(self):
        return self.df

    def best_per_model(self, metric):

        direction = self._get_direction(metric)

        if direction == 1:
            idx = self.df.groupby("model")[metric].idxmax()
        else:
            idx = self.df.groupby("model")[metric].idxmin()

        return self.df.loc[idx]
