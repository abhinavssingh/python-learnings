from abc import abstractmethod

import tensorflow as tf

from .tensorflow_model_wrapper import (
    TensorFlowModelWrapper,
)


class SequentialWrapper(TensorFlowModelWrapper):
    def __init__(self):
        self.model = self.build_model()

    @abstractmethod
    def build_model(self) -> tf.keras.Sequential:
        raise NotImplementedError

    def get_model(self) -> tf.keras.Sequential:
        return self.model

    def summary(self):
        self.model.summary()
