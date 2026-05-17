import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ModelPerformanceVisualizer:
    """
    Utility class for visualizing ML model performance using Plotly
    """

    def __init__(self, results_dict):
        self.results = results_dict

    # ---------------------------------------------------
    # 1. MODEL COMPARISON (BAR CHART)
    # ---------------------------------------------------
    def plot_model_comparison(self):

        data = []

        for model, details in self.results.items():
            if 'metrics' not in details:
                continue

            data.append({
                "Model": model,
                "MSE": details["metrics"]["MSE"],
                "R2": details["metrics"]["R2"]
            })

        df = pd.DataFrame(data)

        fig = go.Figure()

        # ✅ R2 (Left Axis)
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df["R2"],
            name="R2 Score",
            yaxis="y1"
        ))

        # ✅ MSE (Right Axis)
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df["MSE"],
            name="MSE",
            yaxis="y2"
        ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis=dict(title="Model"),

            # Left Y-axis (R2)
            yaxis=dict(
                title="R2 Score",
                range=[0, 1]
            ),

            # Right Y-axis (MSE)
            yaxis2=dict(
                title="MSE",
                overlaying="y",
                side="right"
            ),

            barmode='group'
        )

        return fig

    # ---------------------------------------------------
    # 2. ACTUAL vs PREDICTED
    # ---------------------------------------------------

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name):

        df = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred
        })

        fig = px.scatter(
            df,
            x="Actual",
            y="Predicted",
            title=f"{model_name} - Actual vs Predicted"
        )

        # reference line
        fig.add_trace(go.Scatter(
            x=[df["Actual"].min(), df["Actual"].max()],
            y=[df["Actual"].min(), df["Actual"].max()],
            mode='lines',
            name='Perfect Prediction'
        ))

        return fig

    # ---------------------------------------------------
    # 3. RESIDUAL PLOT
    # ---------------------------------------------------
    def plot_residuals(self, y_true, y_pred, model_name):

        residuals = y_true - y_pred

        df = pd.DataFrame({
            "Predicted": y_pred,
            "Residuals": residuals
        })

        fig = px.scatter(
            df,
            x="Predicted",
            y="Residuals",
            title=f"{model_name} - Residual Plot"
        )

        fig.add_hline(y=0)

        return fig

    # ---------------------------------------------------
    # 4. ERROR DISTRIBUTION
    # ---------------------------------------------------
    def plot_error_distribution(self, y_true, y_pred, model_name):

        residuals = y_true - y_pred

        fig = px.histogram(
            residuals,
            nbins=50,
            title=f"{model_name} - Error Distribution"
        )

        return fig
