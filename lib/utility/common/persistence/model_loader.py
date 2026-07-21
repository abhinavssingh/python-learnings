from abc import ABC, abstractmethod


class ModelLoader(ABC):

    @abstractmethod
    def load(self, filepath):
        pass
