from typing import Optional

from .BaseComparator import BaseComparator


class UnsupervisedModelComparator(BaseComparator):

    DEFAULT_METRIC = "silhouette_score"

    def rank(self, metric: str = None, ascending: Optional[bool] = None):
        metric = metric or self.DEFAULT_METRIC
        return super().rank(metric, ascending)

    def best_model(self, metric: str = None):
        metric = metric or self.DEFAULT_METRIC
        return super().best_model(metric)
