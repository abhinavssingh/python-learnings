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
import json
import uuid

import matplotlib.pyplot as plt
from plotly.utils import PlotlyJSONEncoder

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

    def plot_to_html(self, plot_obj):
        """
        Convert supported plot objects to HTML.

        Returns:
            tuple[str, str | None]:
                (html, plotly_var) for Plotly
                (html, None) for other libraries
        """

        # ✅ Plotly (special handling)
        if hasattr(plot_obj, "to_html") and hasattr(plot_obj, "to_plotly_json"):
            plotly_var = f"fig_{uuid.uuid4().hex[:8]}"
            div_id = f"plot_{plotly_var}"

            # ✅ Inline chart (Plotly-native rendering)
            plot_html = plot_obj.to_html(
                full_html=False,
                include_plotlyjs=True,
                div_id=div_id,
                config={"responsive": True}
            )

            # ✅ Extract JSON for modal re-render
            fig_json = plot_obj.to_plotly_json()

            # ✅ IMPORTANT: UNESCAPED <script> tag
            plot_html += f"""
            <script>
            var {plotly_var} = {json.dumps(fig_json, cls=PlotlyJSONEncoder)};
            </script>
            """
            return plot_html, plotly_var

        # ✅ Other libraries (Bokeh, Altair, etc.)
        if hasattr(plot_obj, "to_html"):
            try:
                return plot_obj.to_html(full_html=False), None
            except TypeError:
                return plot_obj.to_html(), None

        # ✅ Matplotlib Figure
        if isinstance(plot_obj, plt.Figure):
            return self._matplotlib_figure_to_html(plot_obj), None

        # ✅ Matplotlib Axes
        if hasattr(plot_obj, "get_figure"):
            return self._matplotlib_figure_to_html(plot_obj.get_figure()), None

        raise TypeError(
            f"Unsupported plot object type: {type(plot_obj).__name__}"
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
        plot_html, plotly_var = self.plot_to_html(plot_obj)
        return self.components.chart_card(title, plot_html, plotly_var)
