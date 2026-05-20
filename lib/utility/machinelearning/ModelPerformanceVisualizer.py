import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ModelPerformanceVisualizer:
    """
    Utility class for visualizing ML model performance using Plotly
    """

    def __init__(self, results=None):
        """
        results : dictionary from LinearModelUtility (optional)
        """
        self.results = results if results is not None else {}

    # ---------------------------------------------------
    # 1. MODEL COMPARISON FOR ALL
    # ---------------------------------------------------
    def plot_all_model_comparison(self):
        """
        Automatically compares ALL models available in results
        """

        data = []

        for model in self.results.keys():

            res = self.get_flat_result(model)

            if res is None:
                continue

            # ✅ skip models without metrics (e.g., gridsearch-only entries)
            if "MSE" not in res or "R2" not in res:
                continue

            data.append({
                "Model": model,
                "MSE": res.get("MSE"),
                "R2": res.get("R2")
            })

        df = pd.DataFrame(data)

        if df.empty:
            print("No valid model results found.")
            return None

        fig = go.Figure()

        # ✅ R2 Bars
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df["R2"],
            name="R2 Score"
        ))

        # ✅ MSE Bars (secondary axis)
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df["MSE"],
            name="MSE",
            yaxis="y2"
        ))

        fig.update_layout(
            title="All Models Comparison",
            xaxis=dict(title="Model"),

            yaxis=dict(
                title="R2 Score",
                range=[0, 1]
            ),

            yaxis2=dict(
                title="MSE",
                overlaying="y",
                side="right"
            ),

            barmode='group'
        )

        return fig

    # ---------------------------------------------------
    # 2. MODEL COMPARISON (SELECTED MODELS)
    # ---------------------------------------------------

    def plot_model_comparison(self, model_list, results=None):

        results = results or self.results
        data = []

        for model in model_list:
            res = self.get_flat_result(model, results)

            if res is None:
                continue

            data.append({
                "Model": model,
                "MSE": res.get("MSE"),
                "R2": res.get("R2")
            })

        df = pd.DataFrame(data)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df["R2"],
            name="R2 Score"
        ))

        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df["MSE"],
            name="MSE",
            yaxis="y2"
        ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis=dict(title="Model"),
            yaxis=dict(title="R2 Score", range=[0, 1]),
            yaxis2=dict(title="MSE", overlaying="y", side="right"),
            barmode='group'
        )

        return fig

    # ---------------------------------------------------
    # 3. ACTUAL vs PREDICTED
    # ---------------------------------------------------
    def plot_actual_vs_predicted(self, model_name, results=None):

        results = results or self.results
        res = self.get_flat_result(model_name, results)

        if res is None:
            return None

        df = pd.DataFrame({
            "Actual": res["y_true"],
            "Predicted": res["y_pred"]
        })

        fig = px.scatter(
            df,
            x="Actual",
            y="Predicted",
            title=f"{model_name} - Actual vs Predicted"
        )

        fig.add_trace(go.Scatter(
            x=[df["Actual"].min(), df["Actual"].max()],
            y=[df["Actual"].min(), df["Actual"].max()],
            mode='lines',
            name='Perfect Prediction'
        ))

        return fig

    # ---------------------------------------------------
    # 4. RESIDUAL PLOT
    # ---------------------------------------------------
    def plot_residuals(self, model_name, results=None):

        results = results or self.results
        res = self.get_flat_result(model_name, results)

        if res is None:
            return None

        residuals = res["y_true"] - res["y_pred"]

        df = pd.DataFrame({
            "Predicted": res["y_pred"],
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
    # 5. TOTAL ERROR DISTRIBUTION FOR SELECTED MODEL
    # ---------------------------------------------------

    def plot_total_error(self, model_name, mode="absolute", results=None):
        """
        Plot total error as a single bar

        mode:
            'absolute' → sum(|y_true - y_pred|)
            'squared' → sum((y_true - y_pred)^2)
        """

        results = results or self.results
        res = self.get_flat_result(model_name, results)

        if res is None:
            return None

        y_true = res["y_true"]
        y_pred = res["y_pred"]

        if mode == "absolute":
            total_error = abs(y_true - y_pred).sum()
            title = "Total Absolute Error"

        elif mode == "squared":
            total_error = ((y_true - y_pred) ** 2).sum()
            title = "Total Squared Error (SSE)"

        else:
            print("Invalid mode. Use 'absolute' or 'squared'")
            return None

        df = pd.DataFrame({
            "Model": [model_name],
            "Total Error": [total_error]
        })

        fig = px.bar(
            df,
            x="Model",
            y="Total Error",
            title=f"{model_name} - {title}"
        )

        return fig

    # ---------------------------------------------------
    # 6. TOTAL ERROR DISTRIBUTION FOR ALL MODEL
    # ---------------------------------------------------

    def plot_total_error_all(self, mode="absolute"):

        data = []

        for model in self.results.keys():

            res = self.get_flat_result(model)

            if res is None:
                continue

            # ✅ FIX: ensure prediction data exists
            if "y_true" not in res or "y_pred" not in res:
                continue

            y_true = res["y_true"]
            y_pred = res["y_pred"]

            if mode == "absolute":
                total_error = abs(y_true - y_pred).sum()

            elif mode == "squared":
                total_error = ((y_true - y_pred) ** 2).sum()

            else:
                print("Invalid mode")
                return None

            data.append({
                "Model": model,
                "Total Error": total_error
            })

        if not data:
            print("No models with prediction data available.")
            return None

        df = pd.DataFrame(data).sort_values(by="Total Error")

        fig = px.bar(
            df,
            x="Model",
            y="Total Error",
            title=f"Total Error Comparison ({mode})"
        )

        return fig
    # ---------------------------------------------------
    # 7. HELPER: FLATTEN RESULT
    # ---------------------------------------------------

    def get_flat_result(self, model_name, results=None):

        results = results or self.results

        if model_name not in results:
            print(f"No results found for model: {model_name}")
            return None

        res = results[model_name]

        flat_result = {}

        for key, value in res.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_result[sub_key] = sub_value
            else:
                flat_result[key] = value

        return flat_result
