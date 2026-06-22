import pandas as pd
import plotly.express as px


def _resolve_group_col(df):
    return "experiment" if "experiment" in df.columns else "model"


def plot_all(results):

    df = pd.DataFrame(results)

    if df.empty:
        return {
            "r2": px.bar(title="No data available"),
            "rmse": px.bar(title="No data available")
        }

    group_col = _resolve_group_col(df)

    # ✅ R2 Plot
    if "R2" not in df.columns:
        fig_r2 = px.bar(title="R2 not available")
    else:
        fig_r2 = px.bar(
            df,
            x=group_col,
            y="R2",
            color=group_col,
            title=f"R2 Score by {group_col}"
        )

    # ✅ RMSE Plot
    if "RMSE" not in df.columns:
        fig_rmse = px.bar(title="RMSE not available")
    else:
        fig_rmse = px.bar(
            df,
            x=group_col,
            y="RMSE",
            color=group_col,
            title=f"RMSE by {group_col}"
        )

    return {
        "r2": fig_r2,
        "rmse": fig_rmse
    }
