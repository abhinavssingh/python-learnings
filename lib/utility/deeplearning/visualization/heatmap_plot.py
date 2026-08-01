import pandas as pd
import plotly.graph_objects as go


def create_plot(corr_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=20, t=70, b=80),
    )

    return fig


class HeatmapPlot:
    @staticmethod
    def create_plot(corr_df: pd.DataFrame, title: str) -> go.Figure:
        return create_plot(corr_df, title)
