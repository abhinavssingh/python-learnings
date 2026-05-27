import numpy as np
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

    def plot_hyperparameter_surface_3d(self, results_df):

        df = results_df.copy()

        # ✅ auto-detect param columns
        param_cols = [col for col in df.columns if col.startswith("param_")]

        if len(param_cols) < 2:
            print("Not enough hyperparameter columns for 3D plot")

            # ✅ fallback plot instead of returning None
            return px.scatter(
                df,
                x="MSE",
                y="R2",
                color="model",
                title="Fallback: Model Performance Scatter"
            )

        x_col = param_cols[0]
        y_col = param_cols[1]

        pivot_df = df.pivot_table(
            index=y_col,
            columns=x_col,
            values="R2"
        )

        fig = go.Figure(data=[go.Surface(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index
        )])

        fig.update_layout(
            title=f"Hyperparameter Surface ({x_col} vs {y_col})",
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
