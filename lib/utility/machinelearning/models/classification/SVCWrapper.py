from sklearn.svm import SVC

from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class SVCWrapper(ClassificationModelWrapper):
    """
    Wrapper for Support Vector Classifier.
    """

    def __init__(self):
        super().__init__(SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42))

        self.family = "kernel"

        # ✅ Tuning support
        self.param_grid = {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }
