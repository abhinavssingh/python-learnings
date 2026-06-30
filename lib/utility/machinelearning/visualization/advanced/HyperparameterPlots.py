import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.shared.DataCleaner import DataCleaner


from lib.utility.machinelearning._logging import ExceptionLoggingMixin


class HyperparameterPlots(ExceptionLoggingMixin):

    def plot_3d_surface(self, df, x_param=None, y_param=None, metric=None):
        """
        Plot hyperparameter space.
        Automatically adapts to:
        - 3D (2 params)
        - 2D (single param)
        """

        df = df.copy()

        # ✅ Use tuned rows only
        if "type" in df.columns:
            df = df[df["type"] == "tuned"]

        if df.empty:
            return px.scatter(title="No tuned results available")

        # ✅ Auto-detect metric
        if not metric:
            if "score" in df.columns:
                metric = "score"
            elif "R2" in df.columns:
                metric = "R2"
            else:
                return px.scatter(title="No suitable metric found")

        # ✅ Detect hyperparameters
        param_cols = [c for c in df.columns if c.startswith("param_")]

        if not param_cols:
            return px.scatter(title="No hyperparameter columns found")

        # ✅ Select params if not provided
        if not x_param:
            x_param = param_cols[0]

        if not y_param and len(param_cols) > 1:
            y_param = param_cols[1]

        # ✅ Validate required columns
        required_cols = [x_param, metric]
        if y_param:
            required_cols.append(y_param)

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return px.scatter(title=f"Missing columns: {missing}")

        # ✅ Clean data
        cleaner = DataCleaner(df)
        df = cleaner.clean(required_cols)

        if df.empty:
            return px.scatter(title="No valid data after cleaning")

        # ✅ Use experiment if available
        color_col = "experiment" if "experiment" in df.columns else "model"

        # ---------------------------------------------------
        # ✅ 1 PARAM → 2D SCATTER
        # ---------------------------------------------------
        if not y_param:
            return px.scatter(
                df,
                x=x_param,
                y=metric,
                color=color_col,
                title=f"{metric} vs {x_param}"
            )

        # ---------------------------------------------------
        # ✅ 2 PARAMS → 3D SCATTER
        # ---------------------------------------------------
        fig = go.Figure(data=[
            go.Scatter3d(
                x=df[x_param],
                y=df[y_param],
                z=df[metric],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df[metric],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df[color_col] if color_col in df.columns else None,
                name="Hyperparameter Space"
            )
        ])

        fig.update_layout(
            title="Hyperparameter Optimization Surface",
            scene=dict(
                xaxis_title=x_param,
                yaxis_title=y_param,
                zaxis_title=metric
            )
        )

        return fig

