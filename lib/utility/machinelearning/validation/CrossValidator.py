import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate


class CrossValidator:

    @staticmethod
    def run(
        estimator,
        X,
        y,
        problem_type,
        strategy,
        n_splits,
        shuffle,
        random_state,
        scoring
    ):

        if strategy == "holdout":
            return None

        if problem_type in [
            "binary",
            "multiclass",
            "multilabel-indicator"
        ]:

            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        else:

            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        scores = cross_validate(estimator=estimator, X=X, y=y, cv=cv,
                                scoring=scoring, n_jobs=8, return_train_score=False)

        results = {}

        for metric_name, values in scores.items():

            if not metric_name.startswith("test_"):
                continue

            metric = metric_name.replace(
                "test_",
                ""
            )

            results[metric] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4)
            }

        return results
