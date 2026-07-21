from lib.utility.common.metrics.regression_metrics import RegressionMetrics


class RegressionEvaluator:
    def evaluate(self, y_true, y_pred) -> dict:
        return RegressionMetrics.calculate(y_true, y_pred)
