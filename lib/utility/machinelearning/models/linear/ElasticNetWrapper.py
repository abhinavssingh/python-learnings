from sklearn.linear_model import ElasticNet

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class ElasticNetWrapper(RegressionModelWrapper):

    def __init__(self):
        super().__init__(ElasticNet())
