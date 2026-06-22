from typing import Optional

from .BaseComparator import BaseComparator


class ClassificationModelComparator(BaseComparator):

    DEFAULT_METRIC = "f1_weighted"   # ✅ better than accuracy

    # ✅ FIXED signature (matches BaseComparator)
    def rank(self, metric: str = None, ascending: Optional[bool] = None):

        metric = metric or self.DEFAULT_METRIC

        # ✅ classification-specific default (higher is better)
        if ascending is None:
            ascending = False

        return super().rank(metric, ascending)

    def best_model(self, metric: str = None):

        metric = metric or self.DEFAULT_METRIC
        return super().best_model(metric)

    def compare(self, metric: str = None):

        metric = metric or self.DEFAULT_METRIC
        return self.rank(metric)

    def best_per_model(self, metric: str = None):

        metric = metric or self.DEFAULT_METRIC
        return super().best_per_model(metric)

    # ✅ Filtering utilities
    def filter_mode(self, mode):
        return self.df[self.df["mode"] == mode]

    def filter_type(self, type_):
        return self.df[self.df["type"] == type_]
