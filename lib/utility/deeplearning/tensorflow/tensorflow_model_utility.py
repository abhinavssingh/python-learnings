from lib.utility.common.base_experiment_utility import BaseExperimentUtility
from lib.utility.deeplearning.tensorflow.training.trainer import Trainer


class TensorFlowModelUtility(BaseExperimentUtility):

    def __init__(self, model_wrapper, config):
        super().__init__()

        self.model_wrapper = model_wrapper
        self.model = model_wrapper.get_model()

        self.config = config

        self.trainer = Trainer(
            model=self.model,
            config=config
        )

        self.history = None

    def compile(self, loss, metrics=None):
        self.trainer.compile_model(
            loss=loss,
            metrics=metrics
        )

    def train(self, X_train, y_train):
        self.history = self.trainer.fit(
            X_train,
            y_train
        )

        return self.history

    def evaluate(self, X_test, y_test):
        return self.trainer.evaluate(
            X_test,
            y_test
        )

    def predict(self, X):
        return self.trainer.predict(X)

    def plot_results(self):
        pass
