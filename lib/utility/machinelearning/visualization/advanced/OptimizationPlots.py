import numpy as np
import pandas as pd
import plotly.express as px

from lib.utility.machinelearning.visualization.core.DataCleaner import DataCleaner


class OptimizationPlots:

    def plot_optimization_animation(self, df, x_metric, y_metric):
        cleaner = DataCleaner(df)
        df = cleaner.clean([x_metric, y_metric])

        if df.empty:
            return px.scatter(title="No valid data")

        df["iteration"] = range(len(df))

        return px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            animation_frame="iteration",
            color="model",
            size=y_metric,
            title="Optimization Progress"
        )

    def plot_gridsearch_animation(self, df, model=None, mode=None):

        df = df.copy()

        # ✅ filter tuned rows only
        if "type" in df.columns:
            df = df[df["type"] == "tuned"]

        if model:
            df = df[df["model"] == model]

        if mode:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.scatter(title="No gridsearch data found")

        rows = []

        # ✅ Extract param data
        for _, row in df.iterrows():

            # depends on structure
            alphas = row.get("param_model__alpha")
            scores = row.get("mean_test_score")

            # ✅ handle lists/numpy
            if not isinstance(alphas, (list, tuple, np.ndarray)):
                continue

            if not isinstance(scores, (list, tuple, np.ndarray)):
                continue

            if len(alphas) != len(scores):
                continue

            for i in range(len(alphas)):
                try:
                    rows.append({
                        "model": row["model"],
                        "iteration": i,
                        "alpha": float(alphas[i]),
                        "score": float(scores[i])
                    })
                except BaseException:
                    continue

        plot_df = pd.DataFrame(rows)

        if plot_df.empty:
            return px.scatter(title="No valid gridsearch data")

        # ✅ cumulative best
        plot_df = plot_df.sort_values(["model", "iteration"])
        plot_df["best_score"] = plot_df.groupby("model")["score"].cummax()

        fig = px.scatter(
            plot_df,
            x="alpha",
            y="score",
            animation_frame="iteration",
            color="model",
            size="score",
            title="GridSearch Optimization"
        )

        fig.add_scatter(
            x=plot_df["alpha"],
            y=plot_df["best_score"],
            mode="lines",
            name="Best So Far"
        )

        fig.update_xaxes(type="log")

        return fig
