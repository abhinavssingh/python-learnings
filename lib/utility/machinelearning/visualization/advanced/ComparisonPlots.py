import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.shared.DataCleaner import DataCleaner


class ComparisonPlots:
    """
    All model comparison, ranking, and highlighting plots.
    Now fully experiment-aware.
    """

    # ---------------------------------------------------
    # INTERNAL UTIL
    # ---------------------------------------------------
    def _resolve_group_col(self, df):
        return "experiment" if "experiment" in df.columns else "model"

    def _resolve_metric(self, df, metric):
        if metric in df.columns:
            return metric
        if "score" in df.columns:
            return "score"
        if "R2" in df.columns:
            return "R2"
        return None

    # ----------------------------
    # 1. ALL MODEL COMPARISON
    # ----------------------------
    def plot_all_model_comparison(self, df, metric1="R2", metric2="MSE", mode=None):

        metric1 = self._resolve_metric(df, metric1)
        metric2 = self._resolve_metric(df, metric2)

        if not metric1 or not metric2:
            return px.bar(title="Metrics not available")

        cleaner = DataCleaner(df)
        df = cleaner.clean([metric1, metric2])

        if mode:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.bar(title="No valid data")

        group_col = self._resolve_group_col(df)

        fig = go.Figure()

        fig.add_bar(x=df[group_col], y=df[metric1], name=metric1)
        fig.add_bar(x=df[group_col], y=df[metric2], name=metric2, yaxis="y2")

        fig.update_layout(
            title=f"Model Comparison ({mode})",
            yaxis=dict(title=metric1),
            yaxis2=dict(title=metric2, overlaying="y", side="right"),
            barmode="group"
        )

        return fig

    # ----------------------------
    # 2. SELECTED MODEL COMPARISON
    # ----------------------------
    def plot_model_comparison(self, df, model_list, metric="R2", mode=None):

        metric = self._resolve_metric(df, metric)
        if not metric:
            return px.bar(title="Metric not available")

        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        df = df[df["model"].isin(model_list)]

        if mode:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.bar(title="No valid data")

        group_col = self._resolve_group_col(df)

        return px.bar(
            df,
            x=group_col,
            y=metric,
            color=group_col,
            title="Selected Model Comparison"
        )

    # ----------------------------
    # 3. BEST MODEL (HIGHLIGHT)
    # ----------------------------
    def plot_best_model_highlight(self, df, metric="R2"):

        metric = self._resolve_metric(df, metric)
        if not metric:
            return px.bar(title="Metric not available")

        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        if df.empty:
            return px.bar(title="No valid data")

        best_idx = df[metric].idxmax()

        df["highlight"] = "Other"
        df.loc[best_idx, "highlight"] = "Best"

        group_col = self._resolve_group_col(df)

        return px.bar(
            df,
            x=group_col,
            y=metric,
            color="highlight",
            title=f"Best Model Highlight ({metric})"
        )

    # ----------------------------
    # 4. BEST PER MODEL
    # ----------------------------
    def plot_best_per_model(self, df, metric="R2"):

        metric = self._resolve_metric(df, metric)
        if not metric:
            return px.bar(title="Metric not available")

        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        if df.empty:
            return px.bar(title="No valid data")

        idx = df.groupby("model")[metric].idxmax()
        df_best = df.loc[idx]

        return px.bar(
            df_best,
            x="model",
            y=metric,
            color="model",
            title="Best Configuration per Model"
        )

    # ----------------------------
    # 5. MODE COMPARISON
    # ----------------------------
    def plot_mode_comparison(self, df, metric="R2"):

        metric = self._resolve_metric(df, metric)
        if not metric:
            return px.bar(title="Metric not available")

        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        if df.empty:
            return px.bar(title="No valid data")

        group_col = self._resolve_group_col(df)

        return px.bar(
            df,
            x=group_col,
            y=metric,
            color="mode",
            barmode="group",
            title="Mode Comparison"
        )

    # ----------------------------
    # 6. PREPROCESSING IMPACT
    # ----------------------------
    def plot_preprocessing_impact(self, df, x="MSE", y="R2"):

        cleaner = DataCleaner(df)
        df = cleaner.clean([x, y])

        if df.empty:
            return px.scatter(title="No valid data")

        color_col = self._resolve_group_col(df)
        symbol_col = "imputer" if "imputer" in df.columns else None

        return px.scatter(
            df,
            x=x,
            y=y,
            color=color_col,
            symbol=symbol_col,
            title="Preprocessing Impact"
        )

    # ----------------------------
    # 7. MODEL RANKING
    # ----------------------------
    def plot_model_ranking(self, df, metric="R2"):

        metric = self._resolve_metric(df, metric)
        if not metric:
            return px.bar(title="Metric not available")

        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        if df.empty:
            return px.bar(title="No valid data")

        df = df.sort_values(metric, ascending=False)
        df["rank"] = range(1, len(df) + 1)

        group_col = self._resolve_group_col(df)

        return px.bar(
            df,
            x=group_col,
            y="rank",
            title="Model Ranking (Lower is Better)",
            color=group_col
        )
