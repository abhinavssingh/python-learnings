import pandas as pd
import plotly.graph_objects as go


def create_plot(values: pd.Series, title: str, x_title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=40,
                marker_color="#117A65",
                opacity=0.85,
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Count",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


class HistogramPlot:
    @staticmethod
    def create_plot(values: pd.Series, title: str, x_title: str) -> go.Figure:
        return create_plot(values, title, x_title)
