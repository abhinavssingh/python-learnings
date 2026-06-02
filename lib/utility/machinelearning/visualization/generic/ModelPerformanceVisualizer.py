import plotly.express as px

from lib.utility.machinelearning.visualization.core.DataCleaner import DataCleaner
from lib.utility.machinelearning.visualization.core.MetricResolver import MetricResolver


class ModelPerformanceVisualizer:
    """
    Generic ML visualizer (supports regression, classification, etc.)
    """

    def __init__(self):
        pass

    # ----------------------------
    # FILTER
    # ----------------------------
    def _filter(self, df, model=None, mode=None):
        df = df.copy()

        if model:
            df = df[df["model"] == model]

        if mode:
            df = df[df["mode"] == mode]

        return df

    # ----------------------------
    # GENERIC SCATTER
    # ----------------------------
    def plot_scatter(
        self,
        df,
        x_metric=None,
        y_metric=None,
        size=None,
        color="model",
        mode=None
    ):
        df = self._filter(df, mode=mode)

        resolver = MetricResolver(df)

        if not x_metric or not y_metric:
            x_metric, y_metric = resolver.get_default_metrics()

        if x_metric is None or y_metric is None:
            return px.scatter(title="Not enough metrics")

        # ✅ clean data
        cleaner = DataCleaner(df)
        df = cleaner.clean(required_cols=[x_metric, y_metric])

        # ✅ handle size safely
        if size and size in df.columns:
            df[size] = df[size].fillna(df[size].median())

        if df.empty:
            return px.scatter(title="No valid data")

        return px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            size=size,
            color=color,
            title=f"{y_metric} vs {x_metric}"
        )

    # ----------------------------
    # BAR COMPARISON
    # ----------------------------
    def plot_bar(self, df, metric, group_by="model", mode=None):
        df = self._filter(df, mode=mode)

        cleaner = DataCleaner(df)
        df = cleaner.clean(required_cols=[metric])

        if df.empty:
            return px.bar(title="No valid data")

        return px.bar(
            df,
            x=group_by,
            y=metric,
            color=group_by,
            title=f"{metric} Comparison"
        )

    # ----------------------------
    # BEST MODEL
    # ----------------------------
    def plot_best_model(self, df, metric):
        cleaner = DataCleaner(df)
        df = cleaner.clean(required_cols=[metric])

        if df.empty:
            return px.bar(title="No valid data")

        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]

        fig = px.bar(df, x="model", y=metric, color="model")

        fig.add_annotation(
            x=best_row["model"],
            y=best_row[metric],
            text=f"Best: {best_row['model']}<br>{metric}={best_row[metric]:.4f}",
            showarrow=True
        )

        return fig

    # ----------------------------
    # AUTO PLOT (SMART)
    # ----------------------------

    def auto_plot(self, df):
        resolver = MetricResolver(df)
        x_metric, y_metric = resolver.get_default_metrics()

        return self.plot_scatter(df, x_metric, y_metric)
