import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ModelPerformanceVisualizer:
    """
    Utility class for visualizing ML model performance using Plotly
    """

    def __init__(self):
        pass

    # ---------------------------------------------------
    # 1. MODEL COMPARISON FOR ALL
    # ---------------------------------------------------

    def plot_all_model_comparison(self, results_df, mode="train-test"):
        """
        Compare all models using DataFrame
        """

        df = results_df.copy()

        # ✅ filter mode
        if mode:
            df = df[df["mode"] == mode]

        # ✅ remove duplicates (important based on your screenshot)
        df = df.drop_duplicates(subset=["model", "mode", "type", "imputer", "outlier_handler"])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df["model"],
            y=df["R2"],
            name="R2 Score"
        ))

        fig.add_trace(go.Bar(
            x=df["model"],
            y=df["MSE"],
            name="MSE",
            yaxis="y2"
        ))

        fig.update_layout(
            title=f"Model Comparison ({mode})",
            xaxis_title="Model",
            yaxis=dict(title="R2 Score", range=[0, 1]),
            yaxis2=dict(title="MSE", overlaying="y", side="right"),
            barmode='group'
        )

        return fig

    # ---------------------------------------------------
    # 2. MODEL COMPARISON (SELECTED MODELS)
    # ---------------------------------------------------

    def plot_model_comparison(self, model_list, results_df, mode="train-test"):

        df = results_df.copy()

        df = df[df["model"].isin(model_list)]

        if mode:
            df = df[df["mode"] == mode]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df["model"],
            y=df["R2"],
            name="R2 Score"
        ))

        fig.add_trace(go.Bar(
            x=df["model"],
            y=df["MSE"],
            name="MSE",
            yaxis="y2"
        ))

        fig.update_layout(
            title="Selected Model Comparison",
            yaxis=dict(title="R2 Score", range=[0, 1]),
            yaxis2=dict(title="MSE", overlaying="y", side="right"),
            barmode='group'
        )

        return fig

    # ---------------------------------------------------
    # 3. Best Model
    # ---------------------------------------------------

    def plot_best_models(self, results_df):

        df = results_df.copy()

        # ✅ pick best per model
        idx = df.groupby("model")["R2"].idxmax()
        df_best = df.loc[idx]

        fig = px.bar(
            df_best.sort_values("R2", ascending=False),
            x="model",
            y="R2",
            title="Best Configuration per Model",
            color="model"
        )

        return fig

    # ---------------------------------------------------
    # 4. K-Fold vs Train-Test Comparison
    # ---------------------------------------------------

    def plot_mode_comparison(self, results_df):

        df = results_df.copy()

        fig = px.bar(
            df,
            x="model",
            y="R2",
            color="mode",
            barmode="group",
            title="Train-Test vs K-Fold Comparison"
        )

        return fig

    # ---------------------------------------------------
    # 5. Imputer / Outlier Impact (VERY POWERFUL)
    # ---------------------------------------------------

    def plot_preprocessing_impact(self, results_df):

        df = results_df.copy()

        fig = px.scatter(
            df,
            x="MSE",
            y="R2",
            color="model",
            symbol="imputer",
            title="Impact of Imputer on Performance"
        )

        return fig

    # ---------------------------------------------------
    # 6. Ranking Plot (You already have rank_R2 👑)
    # ---------------------------------------------------

    def plot_model_ranking(self, results_df):

        df = results_df.copy().sort_values("rank_R2")

        fig = px.bar(
            df,
            x="model",
            y="rank_R2",
            title="Model Ranking (Lower is Better)",
            color="model"
        )

        return fig

    # ---------------------------------------------------
    # 7.     Animated Optimization
    # ---------------------------------------------------

    def plot_optimization_animation(self, results_df):

        df = results_df.copy()

        # create step index
        df["iteration"] = range(len(df))

        fig = px.scatter(
            df,
            x="MSE",
            y="R2",
            animation_frame="iteration",
            color="model",
            size="R2",
            title="Model Optimization Progress"
        )

        return fig

    # ---------------------------------------------------
    # 8. Hyperparameter Surface Plot (for Ridge/ElasticNet)
    # ---------------------------------------------------

    def to_scalar(self, x):
        try:
            if isinstance(x, (np.ndarray, np.ma.MaskedArray)):
                return float(np.asarray(x).flatten()[0])
            if isinstance(x, (list, tuple)):
                return float(x[0])
            if hasattr(x, "item"):
                return float(x.item())
            return float(x)
        except BaseException:
            return None

    def plot_hyperparameter_surface_3d(self, results_df, model=None, mode=None):

        df = results_df.copy()

        # ✅ Filter by model and mode if provided
        if model is not None:
            df = df[df["model"] == model]

        if mode is not None:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.scatter(title="No data for selected model/mode")

        # ✅ Get param columns but drop useless ones (all NaNs)
        param_cols = [
            col for col in df.columns
            if col.startswith("param_") and not df[col].isna().all()
        ]

        if len(param_cols) < 2:
            return px.scatter(
                df,
                x="MSE",
                y="R2",
                title="Not enough hyperparameters for 3D plot"
            )

        # ✅ Pick first two valid params
        x_col, y_col = param_cols[:2]

        # ✅ Convert safely
        df[x_col] = df[x_col].apply(self.to_scalar)
        df[y_col] = df[y_col].apply(self.to_scalar)
        df["R2"] = pd.to_numeric(df["R2"], errors="coerce")

        # ✅ Drop invalid rows
        df = df.dropna(subset=[x_col, y_col, "R2"])

        if df.empty:
            return px.scatter(title="No valid data after cleaning")

        # ✅ 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df["R2"],
            mode='markers',
            marker=dict(
                size=6,
                color=df["R2"],
                colorscale="Viridis",
                colorbar=dict(title="R2")
            )
        )])

        fig.update_layout(
            title=f"3D Hyperparameter Surface | Model={model} | Mode={mode}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title="R2"
            )
        )

        return fig

    # ---------------------------------------------------
    # 9. Hyperparameter 3D Scatter (alternative to surface)
    # ---------------------------------------------------

    def plot_hyperparameter_3d_scatter(self, results_df):

        df = results_df.copy()

        fig = px.scatter_3d(
            df,
            x="param_model__alpha",
            y="param_model__l1_ratio",
            z="R2",
            color="R2",
            size="R2",
            title="3D Hyperparameter Optimization"
        )

        return fig

    # ---------------------------------------------------
    # 10. BEST MODEL HIGHLIGHT
    # ---------------------------------------------------

    def plot_best_model_highlight(self, results_df):

        df = results_df.copy()

        # ✅ best row
        best_idx = df["R2"].idxmax()
        best_model = df.loc[best_idx]

        df["color"] = "Other"
        df.loc[best_idx, "color"] = "Best Model"

        fig = px.bar(
            df,
            x="model",
            y="R2",
            color="color",
            title=f"Best Model Highlighted → {best_model['model']}",
            color_discrete_map={
                "Best Model": "red",
                "Other": "blue"
            }
        )

        return fig

    # ---------------------------------------------------
    # 11. BEST PER MODEL HIGHLIGHT
    # ---------------------------------------------------
    def plot_best_per_model_highlight(self, results_df):

        df = results_df.copy()

        # ✅ best per model
        idx = df.groupby("model")["R2"].idxmax()
        df_best = df.loc[idx]

        best_global_idx = df_best["R2"].idxmax()

        df_best["color"] = "Other"
        df_best.loc[best_global_idx, "color"] = "Best Overall"

        fig = px.bar(
            df_best,
            x="model",
            y="R2",
            color="color",
            title="Best Configuration per Model",
            color_discrete_map={
                "Best Overall": "red",
                "Other": "green"
            }
        )

        return fig

    # ---------------------------------------------------
    # 12. BEST MODEL WITH ANNOTATION
    # ---------------------------------------------------
    def plot_best_with_annotation(self, results_df):

        df = results_df.copy()

        best_row = df.loc[df["R2"].idxmax()]

        fig = px.bar(df, x="model", y="R2", color="model")

        fig.add_annotation(
            x=best_row["model"],
            y=best_row["R2"],
            text=f"🏆 Best: {best_row['model']}<br>R2={best_row['R2']:.4f}",
            showarrow=True,
            arrowhead=2
        )

        return fig

    # ---------------------------------------------------
    # 13. GRIDSEARCH ANIMATION (for tuning visualization)
    # ---------------------------------------------------

    def plot_gridsearch_animation(self, results_df, model=None, mode=None):

        df = results_df.copy()

        # ✅ Filter gridsearch rows first
        df = df[df["mode"].str.contains(mode, case=False, na=False)]

        # ✅ Additional user filters
        if model is not None:
            df = df[df["model"] == model]

        if mode is not None:
            df = df[df["mode"] == mode]

        if df.empty:
            return px.scatter(title="No gridsearch rows found for given filters")

        rows = []

        for _, row in df.iterrows():

            alphas = row.get("param_model__alpha")
            scores = row.get("mean_test_score")

            # ✅ Allow list / tuple / numpy array
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
                        "mode": row["mode"],
                        "alpha": float(alphas[i]),
                        "score": float(scores[i]),
                        "iteration": i
                    })
                except Exception:
                    continue

        plot_df = pd.DataFrame(rows)

        if plot_df.empty:
            return px.scatter(title="No valid hyperparameter data")

        # ✅ Sorting + cumulative best
        plot_df = plot_df.sort_values(["model", "iteration"])
        plot_df["best_score"] = plot_df.groupby("model")["score"].cummax()

        fig = px.scatter(
            plot_df,
            x="alpha",
            y="score",
            animation_frame="iteration",
            color="model",
            size="score",
            hover_data=["mode"],  # ✅ shows mode in tooltip
            title=f"GridSearch Optimization | Model={model} | Mode={mode}"
        )

        fig.update_xaxes(type="log")

        # ✅ Best-so-far line
        fig.add_scatter(
            x=plot_df["alpha"],
            y=plot_df["best_score"],
            mode="lines",
            name="Best So Far"
        )

        return fig

    # ---------------------------------------------------
    # 8. HELPER: FLATTEN RESULT
    # ---------------------------------------------------

    def get_flat_result(self, model_name, results, parent_key="", sep="_"):

        if model_name not in results:
            print(f"No results found for model: {model_name}")
            return None

        res = results[model_name]

        def flatten_dict(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k

                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, (list, tuple)):
                    # Optional: handle lists (e.g., CV scores)
                    if all(isinstance(i, (int, float)) for i in v):
                        items.append((new_key + "_mean", np.mean(v)))
                        items.append((new_key + "_std", np.std(v)))
                    else:
                        items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))

            return dict(items)

        flat_result = flatten_dict(res, parent_key, sep)
        return flat_result
