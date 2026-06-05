from sklearn.linear_model import Lasso

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class LassoWrapper(RegressionModelWrapper):

    def __init__(self):
        super().__init__(Lasso())
