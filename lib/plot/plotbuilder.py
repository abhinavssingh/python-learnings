"""
Class-based plot converter that uses ComponentsBuilder for consistent styling.

Supports:
- Plotly, Bokeh, Altair (native to_html() method)
- Matplotlib Figure and Axes objects

Uses ComponentsBuilder components:
- chart_card() for individual plots
- chart_grid_2x2() for 2x2 layouts
- chart_container() for fixed-height containers
"""

import base64
import io

import matplotlib.pyplot as plt

from lib.html.components import ComponentsBuilder


class PlotRenderer:
    """
    Convert any Python plot object to styled HTML using ComponentsBuilder.

    Features:
        - Convert any plot to HTML
        - Wrap plots in styled cards
        - Create responsive grid layouts
        - Add fixed-height containers

    Example:
        renderer = PlotRenderer()
        fig = px.bar(df, x='col', y='val')
        html = renderer.plot_to_card(fig, 'Sales Chart')
    """

    def __init__(self):
        """Initialize with ComponentsBuilder for styling."""
        self.components = ComponentsBuilder()

    def plot_to_html(self, plot_obj) -> str:
        """
        Convert any common Python plot object to HTML.

        Args:
            plot_obj: A plot object from any library
                - Plotly Figure
                - Bokeh Figure
                - Altair Chart
                - Matplotlib Figure
                - Matplotlib Axes

        Returns:
            HTML string representation of the plot

        Raises:
            TypeError: If plot object type is not supported

        Examples:
            # Plotly
            import plotly.express as px
            df = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})
            fig = px.bar(df, x='x', y='y')
            html = renderer.plot_to_html(fig)

            # Matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [4, 5, 6])
            html = renderer.plot_to_html(fig)
        """

        # Check for native to_html() method (Plotly, Bokeh, Altair, etc.)
        if hasattr(plot_obj, "to_html"):
            try:
                return plot_obj.to_html(full_html=False, include_plotlyjs="cdn")
            except TypeError:
                # Some libraries use different parameters
                return plot_obj.to_html()

        # Handle matplotlib Figure
        if isinstance(plot_obj, plt.Figure):
            return self._matplotlib_figure_to_html(plot_obj)

        # Handle matplotlib Axes
        if hasattr(plot_obj, "get_figure"):
            fig = plot_obj.get_figure()
            return self._matplotlib_figure_to_html(fig)

        # Unsupported type
        raise TypeError(
            f"Unsupported plot object type: {type(plot_obj).__name__}. "
            f"Supported: Plotly Figure, Bokeh Figure, Altair Chart, "
            f"Matplotlib Figure, Matplotlib Axes"
        )

    def _matplotlib_figure_to_html(self, fig) -> str:
        """
        Convert matplotlib Figure to HTML base64-encoded PNG image.

        Args:
            fig: matplotlib.figure.Figure object

        Returns:
            HTML string with embedded <img> tag
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Encode as base64
        encoded = base64.b64encode(buffer.getvalue()).decode()

        return f'<img src="data:image/png;base64,{encoded}" style="width:100%; height:auto;"/>'

    def plot_to_card(self, plot_obj, title: str = "") -> str:
        """
        Convert plot to HTML and wrap in a styled card component.

        Uses ComponentsBuilder.chart_card() for consistent styling.

        Args:
            plot_obj: Any supported plot object
            title: Card title

        Returns:
            HTML string with plot wrapped in chart_card component
        """
        plot_html = self.plot_to_html(plot_obj)
        return self.components.chart_card(title, plot_html)

    def plot_to_grid_2x2(self, plots_list: list) -> str:
        """
        Create a 2x2 responsive grid of plots.

        Uses ComponentsBuilder.chart_grid_2x2() for layout.

        Args:
            plots_list: List of tuples (plot_obj, title) up to 4 items

        Returns:
            HTML string with plots in 2x2 grid

        Example:
            plots = [
                (fig1, 'Q1 Sales'),
                (fig2, 'Q2 Sales'),
                (fig3, 'Q3 Sales'),
                (fig4, 'Q4 Sales'),
            ]
            html = renderer.plot_to_grid_2x2(plots)
        """
        if len(plots_list) > 4:
            raise ValueError("Chart grid 2x2 supports maximum 4 plots")

        cards = [
            self.plot_to_card(plot, title)
            for plot, title in plots_list
        ]
        return self.components.chart_grid_2x2(cards)

    def plot_to_container(self, plot_obj, title: str = "", height: int = 300) -> str:
        """
        Wrap plot in a fixed-height container.

        Uses ComponentsBuilder.chart_container() for sizing.

        Args:
            plot_obj: Any supported plot object
            title: Optional chart title
            height: Container height in pixels (default 300)

        Returns:
            HTML string with plot in fixed-height container

        Example:
            html = renderer.plot_to_container(fig, 'Revenue', height=400)
        """
        plot_html = self.plot_to_html(plot_obj)
        container_html = self.components.chart_container(plot_html, height)

        if title:
            return f"""
<div>
    <h3 class="text-base font-semibold mb-2 text-slate-800 dark:text-slate-100">
        {title}
    </h3>
    {container_html}
</div>
"""
        return container_html
