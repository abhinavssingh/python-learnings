from sklearn.linear_model import LinearRegression

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class LinearRegressionWrapper(RegressionModelWrapper):

    def __init__(self):
        super().__init__(LinearRegression())
