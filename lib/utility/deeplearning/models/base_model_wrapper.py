from abc import ABC, abstractmethod


class BaseModelWrapper(ABC):
    """
    Abstract base class for all Deep Learning models.
    """

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def get_model(self):
        pass
