from abc import ABC, abstractmethod


class ModelSaver(ABC):

    @abstractmethod
    def save(self, model, filepath):
        pass
