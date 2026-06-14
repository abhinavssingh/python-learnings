from sklearn.tree import DecisionTreeClassifier

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class DecisionTreeClassifierWrapper(ClassificationModelWrapper):
    """
    Wrapper for Decision Tree classification model.
    """

    def __init__(self):
        super().__init__(DecisionTreeClassifier(max_depth=5, max_features="sqrt", random_state=42))

        self.family = "tree"

        # ✅ Tuning support
        self.param_grid = {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5],
            "model__criterion": ["gini", "entropy"]
        }
