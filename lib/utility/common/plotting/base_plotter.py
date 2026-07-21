from abc import ABC, abstractmethod

import plotly.graph_objects as go


class BasePlotter(ABC):

    @abstractmethod
    def plot(self) -> go.Figure:
        """
        Return a Plotly figure.
        """
        pass

    def save_html(self, filepath: str):
        self.plot().write_html(filepath)

    def save_image(self, filepath: str):
        self.plot().write_image(filepath)
