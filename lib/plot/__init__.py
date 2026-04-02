"""
Plot-to-HTML converter using ComponentsBuilder styling.

Convert any Python plot object (Plotly, Bokeh, Altair, Matplotlib) to styled HTML
using ComponentsBuilder components for consistency with your design system.

Example - Class-based (Recommended):
    from lib.plot import PlotRenderer
    import plotly.express as px
    import pandas as pd

    renderer = PlotRenderer()
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    fig = px.bar(df, x='x', y='y')

    # Single plot in card
    html = renderer.plot_to_card(fig, 'My Chart')

    # Multiple plots in 2x2 grid
    plots = [(fig1, 'Q1'), (fig2, 'Q2'), (fig3, 'Q3'), (fig4, 'Q4')]
    html = renderer.plot_to_grid_2x2(plots)

    # Fixed-height container
    html = renderer.plot_to_container(fig, 'Revenue', height=400)

Example - Function-based (Legacy):
    from lib.plot import plot_to_html, plot_to_card

    html = plot_to_html(fig)
    html = plot_to_card(fig, title='My Plot')
"""

from .plotbuilder import PlotRenderer

__all__ = ["PlotRenderer"]
