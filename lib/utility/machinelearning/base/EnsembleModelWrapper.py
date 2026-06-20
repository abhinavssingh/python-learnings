from lib.utility.machinelearning.base.ClassificationModelWrapper import ClassificationModelWrapper


class EnsembleModelWrapper(ClassificationModelWrapper):

    def __init__(self):
        super().__init__(model=None)

    task = "classification"
    family = "ensemble"
