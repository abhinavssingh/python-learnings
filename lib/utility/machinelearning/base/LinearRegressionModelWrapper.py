from sklearn.pipeline import Pipeline

from lib.utility.machinelearning.base.BaseModelWrapper import BaseModelWrapper
from lib.utility.machinelearning.evaluation.Metrics import Metrics


class RegressionModelWrapper(BaseModelWrapper):
    """
    Base wrapper for all regression models.
    """

    task = "linear regression"
    family = "linear"

    def build_pipeline(self, preprocessor):
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

    def evaluate(self, y_true, y_pred):
        """
        Evaluate regression model performance.
        Can be extended to include residual plots, error distributions, etc.
        """
        return Metrics.regression(
            y_true,
            y_pred,
            # ✅ future-ready (optional flags if you extend Metrics later)
            # include_residuals=True,
            # include_plots=True
        )
