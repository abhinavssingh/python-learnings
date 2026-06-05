from sklearn.linear_model import Ridge

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class RidgeWrapper(RegressionModelWrapper):

    def __init__(self):
        super().__init__(Ridge())
