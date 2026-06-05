import plotly.express as px

from lib.utility.machinelearning.shared.DataCleaner import DataCleaner


class OptimizationPlots:
    """
    Handles optimization / tuning visualizations.
    Now fully compatible with experiment-based architecture.
    """

    # ---------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------
    def _resolve_group_col(self, df):
        return "experiment" if "experiment" in df.columns else "model"

    def _resolve_metric(self, df, metric):
        if metric and metric in df.columns:
            return metric
        if "score" in df.columns:
            return "score"
        if "R2" in df.columns:
            return "R2"
        return None

    # ---------------------------------------------------
    # 1. GENERIC OPTIMIZATION ANIMATION
    # ---------------------------------------------------
    def plot_optimization_animation(self, df, x_metric=None, y_metric=None):

        df = df.copy()

        y_metric = self._resolve_metric(df, y_metric)
        x_metric = x_metric if x_metric in df.columns else None

        if not y_metric:
            return px.scatter(title="No valid metric found")

        # ✅ fallback x-axis
        if not x_metric:
            df["iteration"] = range(len(df))
            x_metric = "iteration"

        cleaner = DataCleaner(df)
        df = cleaner.clean([x_metric, y_metric])

        if df.empty:
            return px.scatter(title="No valid data")

        color_col = self._resolve_group_col(df)

        df["iteration"] = range(len(df))

        return px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            animation_frame="iteration",
            color=color_col,
            size=y_metric,
            title="Optimization Progress"
        )

    # ---------------------------------------------------
    # 2. GRID / RANDOM SEARCH ANIMATION
    # ---------------------------------------------------
    def plot_search_animation(self, df, model=None, mode=None, metric=None):

        df = df.copy()

        # ✅ filter tuned results
        if "type" in df.columns:
            df = df[df["type"] == "tuned"]

        if model:
            df = df[df["model"] == model]

        if mode:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.scatter(title="No tuning data found")

        metric = self._resolve_metric(df, metric)
        if not metric:
            return px.scatter(title="No valid metric found")

        # ✅ detect params
        param_cols = [c for c in df.columns if c.startswith("param_")]

        if not param_cols:
            return px.scatter(title="No hyperparameter data available")

        # ✅ pick first param for animation axis
        x_param = param_cols[0]

        cleaner = DataCleaner(df)
        df = cleaner.clean([x_param, metric])

        if df.empty:
            return px.scatter(title="No valid data after cleaning")

        color_col = self._resolve_group_col(df)

        # ✅ create iteration (sorted improves animation)
        df = df.sort_values(metric, ascending=False)
        df["iteration"] = range(len(df))

        # ✅ cumulative best
        df["best_score"] = df[metric].cummax()

        fig = px.scatter(
            df,
            x=x_param,
            y=metric,
            animation_frame="iteration",
            color=color_col,
            size=metric,
            title="Hyperparameter Optimization Progress"
        )

        # ✅ ✅ Add best trend line (static overlay)
        fig.add_scatter(
            x=df[x_param],
            y=df["best_score"],
            mode="lines",
            name="Best So Far"
        )

        return fig
