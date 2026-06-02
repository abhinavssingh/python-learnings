import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.visualization.core.DataCleaner import DataCleaner


class HyperparameterPlots:

    def plot_3d_surface(self, df, x_param=None, y_param=None, metric="R2"):

        df = df.copy()

        # ✅ Only use tuned results
        if "type" in df.columns:
            df = df[df["type"] == "tuned"]

        if df.empty:
            return px.scatter(title="No tuned results available")

        # ✅ Auto-detect params if not provided
        param_cols = [c for c in df.columns if c.startswith("param_")]

        if not param_cols:
            return px.scatter(title="No hyperparameter columns found")

        if not x_param or not y_param:
            if len(param_cols) < 2:
                return px.scatter(title="Not enough hyperparameters")
            x_param, y_param = param_cols[:2]

        # ✅ Validate existence
        required_cols = [x_param, y_param, metric]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return px.scatter(title=f"Missing columns: {missing}")

        cleaner = DataCleaner(df)
        df = cleaner.clean(required_cols)

        if df.empty:
            return px.scatter(title="No valid data after cleaning")

        return go.Figure(data=[
            go.Scatter3d(
                x=df[x_param],
                y=df[y_param],
                z=df[metric],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df[metric],
                    colorscale='Viridis'
                )
            )
        ])
