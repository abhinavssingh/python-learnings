import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.visualization.core.DataCleaner import DataCleaner


class ComparisonPlots:
    """
    All model comparison, ranking, and highlighting plots.
    """

    # ----------------------------
    # 1. ALL MODEL COMPARISON
    # ----------------------------
    def plot_all_model_comparison(self, df, metric1="R2", metric2="MSE", mode=None):
        cleaner = DataCleaner(df)
        df = cleaner.clean([metric1, metric2])

        if mode:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.bar(title="No valid data")

        fig = go.Figure()

        fig.add_bar(x=df["model"], y=df[metric1], name=metric1)
        fig.add_bar(x=df["model"], y=df[metric2], name=metric2, yaxis="y2")

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
        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        df = df[df["model"].isin(model_list)]

        if mode:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.bar(title="No valid data")

        return px.bar(
            df,
            x="model",
            y=metric,
            color="model",
            title="Selected Model Comparison"
        )

    # ----------------------------
    # 3. BEST MODEL (HIGHLIGHT)
    # ----------------------------
    def plot_best_model_highlight(self, df, metric="R2"):
        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        if df.empty:
            return px.bar(title="No valid data")

        best_idx = df[metric].idxmax()

        df["color"] = "Other"
        df.loc[best_idx, "color"] = "Best Model"

        return px.bar(
            df,
            x="model",
            y=metric,
            color="color",
            title=f"Best Model Highlight ({metric})"
        )

    # ----------------------------
    # 4. BEST PER MODEL
    # ----------------------------
    def plot_best_per_model(self, df, metric="R2"):
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
    # 5. MODE COMPARISON (K-FOLD vs TRAIN-TEST)
    # ----------------------------
    def plot_mode_comparison(self, df, metric="R2"):
        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        if df.empty:
            return px.bar(title="No valid data")

        return px.bar(
            df,
            x="model",
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

        # ✅ SAFE: check if imputer exists
        symbol_col = "imputer" if "imputer" in df.columns else None

        if symbol_col:
            return px.scatter(
                df,
                x=x,
                y=y,
                color="model",
                symbol=symbol_col,
                title="Preprocessing Impact"
            )
        else:
            # ✅ fallback (no symbol)
            return px.scatter(
                df,
                x=x,
                y=y,
                color="model",
                title="Preprocessing Impact (No Imputer Info)"
            )

    # ----------------------------
    # 7. MODEL RANKING
    # ----------------------------

    def plot_model_ranking(self, df, metric="R2"):
        cleaner = DataCleaner(df)
        df = cleaner.clean([metric])

        if df.empty:
            return px.bar(title="No valid data")

        df = df.sort_values(metric, ascending=False)
        df["rank"] = range(1, len(df) + 1)

        return px.bar(
            df,
            x="model",
            y="rank",
            title="Model Ranking (Lower is Better)",
            color="model"
        )
