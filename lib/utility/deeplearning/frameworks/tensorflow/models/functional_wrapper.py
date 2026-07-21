from abc import abstractmethod

import tensorflow as tf

from .tensorflow_model_wrapper import (
    TensorFlowModelWrapper,
)


class FunctionalWrapper(TensorFlowModelWrapper):
    def __init__(self):
        self.model = self.build_model()

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        raise NotImplementedError

    def get_model(self) -> tf.keras.Model:
        return self.model

    def summary(self):
        self.model.summary()
