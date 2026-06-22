import pandas as pd
import plotly.express as px


def _resolve_group_col(df):
    return "experiment" if "experiment" in df.columns else "model"


# ---------------------------------------------------
# ✅ METRIC DISTRIBUTION
# ---------------------------------------------------

def plot_metric_distribution(results, metric):

    df = pd.DataFrame(results)

    if metric not in df.columns:
        return px.histogram(title=f"{metric} not found")

    # ✅ Resolve grouping column (experiment first)
    group_col = _resolve_group_col(df)

    return px.histogram(
        df,
        x=metric,
        color=group_col,
        nbins=20,
        title=f"Distribution of {metric} by {group_col}"
    )


# ---------------------------------------------------
# ✅ RESIDUAL DISTRIBUTION (REGRESSION)
# ---------------------------------------------------

def plot_residual_distribution(y_true, y_pred):

    residuals = y_true - y_pred

    return px.histogram(
        x=residuals,
        nbins=30,
        title="Residual Distribution"
    )


# ---------------------------------------------------
# ✅ CLASS DISTRIBUTION (CLASSIFICATION)
# ---------------------------------------------------

def plot_class_distribution(y):

    return px.histogram(
        x=y,
        title="Class Distribution"
    )


# ---------------------------------------------------
# ✅ CLUSTER SIZE DISTRIBUTION
# ---------------------------------------------------

def plot_cluster_distribution(labels):

    df = pd.DataFrame({"cluster": labels})

    return px.histogram(
        df,
        x="cluster",
        title="Cluster Distribution"
    )
