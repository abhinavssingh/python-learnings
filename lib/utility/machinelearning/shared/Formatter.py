class Formatter:
    """
    Utility class for building consistent experiment names
    across the ML framework.
    """

    @staticmethod
    def build(
        model_name: str,
        mode: str,
        k: int = None,
        imputer=None,
        outlier_handler=None,
        search_type: str = None,
    ) -> str:
        """
        Build a standardized experiment name.

        Parameters:
        - model_name: Name of the model
        - mode: train-test / k-fold / gridsearch / random_search
        - k: folds (if k-fold)
        - imputer: imputer object
        - outlier_handler: outlier handler object
        - search_type: grid / random

        Returns:
        - formatted experiment string
        """

        parts = [model_name]

        # ✅ mode handling
        if mode == "k-fold" and k:
            parts.append(f"kfold={k}")
        elif mode == "train-test":
            parts.append("train-test")
        else:
            parts.append(mode)

        # ✅ search type (optional)
        if search_type:
            parts.append(f"search={search_type}")

        # ✅ preprocessing
        if imputer:
            parts.append(f"imputer={type(imputer).__name__}")

        if outlier_handler:
            parts.append(f"outlier={type(outlier_handler).__name__}")

        return " | ".join(parts)
