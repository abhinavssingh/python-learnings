import pandas as pd
import plotly.graph_objects as go


def create_plot(series: pd.Series, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=series.index.astype(str).tolist(),
                y=series.values.tolist(),
                marker_color="#2E86AB",
                text=series.values.tolist(),
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


class BarChartPlot:
    @staticmethod
    def create_plot(series: pd.Series, title: str, x_title: str, y_title: str) -> go.Figure:
        return create_plot(series, title, x_title, y_title)
