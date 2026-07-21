from abc import ABC, abstractmethod


class BasePersistence(ABC):

    @abstractmethod
    def save(self, model, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
