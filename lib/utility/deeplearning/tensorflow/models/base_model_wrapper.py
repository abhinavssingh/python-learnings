from abc import ABC, abstractmethod

import tensorflow as tf


class BaseModelWrapper(ABC):
    """
    Abstract base class for all Deep Learning models.
    """

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        pass

    @property
    def model_name(self):
        return self.__class__.__name__
