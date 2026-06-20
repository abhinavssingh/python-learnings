from collections import Counter

from imblearn.over_sampling import SMOTE

from .BaseImbalanceHandler import BaseImbalanceHandler


class SMOTEHandler(BaseImbalanceHandler):
    """
    SMOTE wrapper for pipeline integration + audit tracking.
    """

    def __init__(
        self,
        sampling_strategy="auto",
        k_neighbors=5,
        random_state=42,
        **kwargs
    ):
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            **kwargs
        )

        # ✅ NEW: tracking fields
        self.before_counts = None
        self.after_counts = None

    # ✅ REQUIRED for imblearn Pipeline
    def fit_resample(self, X, y):
        """
        Called automatically during pipeline.fit()
        """

        # ✅ Capture BEFORE distribution
        self.before_counts = dict(Counter(y))

        # ✅ Apply SMOTE
        X_res, y_res = self.smote.fit_resample(X, y)

        # ✅ Capture AFTER distribution
        self.after_counts = dict(Counter(y_res))

        return X_res, y_res

    def get_pipeline_step(self):
        """
        Return self instead of raw SMOTE so pipeline calls our fit_resample().
        """
        return ("smote", self)

    # ✅ OPTIONAL: expose info cleanly
    def get_summary(self):
        return {
            "method": "SMOTE",
            "params": self.smote.get_params(),
            "before": self.before_counts,
            "after": self.after_counts
        }
