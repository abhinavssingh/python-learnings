from collections import Counter

from imblearn.over_sampling import SMOTE
from scipy import sparse


class SMOTEResampler:

    @staticmethod
    def resample(
        X,
        y,
        sampling_strategy="auto",
        k_neighbors: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        """Apply SMOTE to training data and return resampled arrays.

        Converts sparse matrices to dense arrays before resampling because
        many imblearn transformers require dense input.
        """

        if sparse.issparse(X):
            X = X.toarray()

        before_counts = dict(Counter(y))

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            **kwargs,
        )

        X_res, y_res = smote.fit_resample(X, y)
        after_counts = dict(Counter(y_res))

        return X_res, y_res, {
            "method": "SMOTE",
            "params": smote.get_params(),
            "before": before_counts,
            "after": after_counts,
        }
