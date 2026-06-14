from sklearn.linear_model import ElasticNet

from lib.utility.machinelearning.base.LinearRegressionModelWrapper import RegressionModelWrapper


class ElasticNetWrapper(RegressionModelWrapper):
    """
    Wrapper for ElasticNet Regression model.
    """

    def __init__(self):
        super().__init__(ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))

        self.family = "linear"
