from .BaseComparator import BaseComparator


class RegressionModelComparator(BaseComparator):
    """
    Comparator for regression models.

    Handles:
    - Ranking
    - Best model selection
    - Per-model best selection
    """

    DEFAULT_METRIC = "R2"

    def rank(self, metric=None, ascending=None):
        metric = metric or self.DEFAULT_METRIC

        if ascending is None:
            ascending = False if metric == "R2" else None

        return super().rank(metric, ascending)

    def best_model(self, metric=None):
        """
        Get best regression model.
        """
        metric = metric or self.DEFAULT_METRIC
        return super().best_model(metric)

    def best_per_model(self, metric=None):
        """
        Best configuration per model (for tuning scenarios)
        """
        metric = metric or self.DEFAULT_METRIC
        return super().best_per_model(metric)

    def compare(self, sort_by=None):
        """
        Return sorted DataFrame for regression models.
        """
        sort_by = sort_by or self.DEFAULT_METRIC
        return self.rank(sort_by)
