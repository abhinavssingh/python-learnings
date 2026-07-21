from abc import abstractmethod

import tensorflow as tf

from lib.utility.deeplearning.abstractions.base_model_wrapper import (
    BaseModelWrapper,
)


class TensorFlowModelWrapper(BaseModelWrapper[tf.keras.Model]):
    """
    Base wrapper for all TensorFlow models.
    """

    model: tf.keras.Model | None = None

    @property
    def model_name(self) -> str:
        return self.model.name if self.model is not None else self.__class__.__name__

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        raise NotImplementedError

    def get_model(self) -> tf.keras.Model:
        if self.model is None:
            self.model = self.build_model()

        return self.model

    def summary(self):
        self.get_model().summary()
