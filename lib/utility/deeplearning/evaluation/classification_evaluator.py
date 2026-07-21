from lib.utility.common.metrics.classification_metrics import ClassificationMetrics


class ClassificationEvaluator:
    def evaluate(self, y_true, y_pred) -> dict:
        return ClassificationMetrics.calculate(y_true, y_pred)
