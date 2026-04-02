"""
plotutility.py

Backend-agnostic heatmap construction utility.

This module provides a single, consistent API for creating heatmaps
from already-aggregated categorical data, using different plotting
libraries (Seaborn or Plotly).

Responsibilities:
- Pivot aggregated data into heatmap matrix
- Dispatch rendering to chosen backend
- Return plot object (Figure), NOT HTML

This module does NOT:
- Aggregate raw data
- Convert plots to HTML
- Handle layout, cards, or grids
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


class PlotUtility:
    """
    Utility class for creating heatmaps using different plotting backends.

    Supported backends:
    - 'seaborn' : Static matplotlib/seaborn heatmap
    - 'plotly'  : Interactive Plotly heatmap (native HTML capable)

    The input DataFrame MUST already be aggregated (e.g. via groupby + size).
    """

    # ======================================================
    # Public API
    # ======================================================
    @staticmethod
    def plot_heatmap(
        df,
        *,
        index: str,
        columns: str,
        values: str = "Count",
        title: str | None = None,
        backend: str = "seaborn",
        figsize: tuple[int, int] = (8, 5),
        cmap: str = "Blues",
        template: str = "plotly_dark",
        height: int = 450,
        show_values: bool = True,
    ):
        """
        Create a heatmap using the specified plotting backend.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregated DataFrame containing categorical counts.
        index : str
            Column to use for heatmap Y-axis.
        columns : str
            Column to use for heatmap X-axis.
        values : str, default "Count"
            Column containing numeric values.
        title : str, optional
            Plot title.
        backend : {"seaborn", "plotly"}, default "seaborn"
            Plotting backend to use.
        figsize : tuple, default (8, 5)
            Figure size for seaborn backend.
        cmap : str, default "Blues"
            Colormap for seaborn heatmap.
        template : str, default "plotly_dark"
            Plotly template for plotly backend.
        height : int, default 450
            Plot height for plotly backend.
        show_values : bool, default True
            Whether to display numeric values on heatmap cells.

        Returns
        -------
        matplotlib.figure.Figure | plotly.graph_objs.Figure
            Heatmap figure object.
        """

        # ✅ Pivot data into matrix form
        pivot_df = (
            df
            .pivot(index=index, columns=columns, values=values)
            .fillna(0)
        )

        # ✅ Backend dispatch
        if backend == "seaborn":
            return PlotUtility._plot_seaborn_heatmap(
                pivot_df,
                index=index,
                columns=columns,
                title=title,
                cmap=cmap,
                figsize=figsize,
                show_values=show_values,
            )

        elif backend == "plotly":
            return PlotUtility._plot_plotly_heatmap(
                pivot_df,
                index=index,
                columns=columns,
                title=title,
                template=template,
                height=height,
                show_values=show_values,
            )

        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                "Supported: 'seaborn', 'plotly'."
            )

    # ======================================================
    # Seaborn backend (Image → HTML)
    # ======================================================
    @staticmethod
    def _plot_seaborn_heatmap(
        pivot_df,
        *,
        index: str,
        columns: str,
        title: str | None,
        cmap: str,
        figsize: tuple[int, int],
        show_values: bool,
    ):
        """
        Create a seaborn/matplotlib heatmap.

        Returns a matplotlib Figure suitable for image-to-HTML conversion.
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            pivot_df,
            annot=show_values,
            fmt=".0f",
            cmap=cmap,
            ax=ax,
        )

        ax.set_xlabel(columns)
        ax.set_ylabel(index)

        if title:
            ax.set_title(title)

        return fig

    # ======================================================
    # Plotly backend (Native HTML)
    # ======================================================
    @staticmethod
    def _plot_plotly_heatmap(
        pivot_df,
        *,
        index: str,
        columns: str,
        title: str | None,
        template: str,
        height: int,
        show_values: bool,
    ):
        """
        Create a Plotly heatmap.

        Returns a Plotly Figure that can emit native HTML.
        """
        fig = px.imshow(
            pivot_df,
            text_auto=show_values,
            color_continuous_scale="Viridis",
            labels={
                "x": columns,
                "y": index,
                "color": "Count",
            },
            title=title,
        )

        fig.update_layout(
            template=template,
            height=height,
        )

        return fig
