from abc import abstractmethod

import tensorflow as tf

from .base_model_wrapper import BaseModelWrapper


class FunctionalWrapper(BaseModelWrapper):
    """
    Base wrapper for TensorFlow Functional API models.
    """

    def __init__(self):
        self.model = self.build_model()

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """
        Build and return a TensorFlow model.
        """
        raise NotImplementedError

    def get_model(self) -> tf.keras.Model:
        return self.model

    def summary(self):
        self.model.summary()
