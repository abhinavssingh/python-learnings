import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.evaluation.MetricResolver import MetricResolver
from lib.utility.machinelearning.shared.DataCleaner import DataCleaner

# ---------------------------------------------------
# ✅ INTERNAL HELPERS
# ---------------------------------------------------


def _resolve_group_col(df):
    return "experiment" if "experiment" in df.columns else "model"


def _get_metric(df, metric, task=None):
    if metric and metric in df.columns:
        return metric

    if task:
        return MetricResolver.get_best_metric(task)

    # fallback
    numeric_cols = df.select_dtypes("number").columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def _get_direction(metric):
    return MetricResolver.get_direction(metric)


# ---------------------------------------------------
# ✅ 1. ALL MODEL COMPARISON
# ---------------------------------------------------

def plot_all_model_comparison(df, metric1=None, metric2=None, mode=None, task=None):

    metric1 = _get_metric(df, metric1, task)
    metric2 = _get_metric(df, metric2, task)

    if not metric1 or not metric2:
        return px.bar(title="Metrics not available")

    cleaner = DataCleaner(df)
    df = cleaner.clean([metric1, metric2])

    if mode:
        df = df[df["mode"] == mode]

    if df.empty:
        return px.bar(title="No valid data")

    group_col = _resolve_group_col(df)

    fig = go.Figure()

    fig.add_bar(x=df[group_col], y=df[metric1], name=metric1)
    fig.add_bar(x=df[group_col], y=df[metric2], name=metric2, yaxis="y2")

    fig.update_layout(
        title=f"Model Comparison ({metric1} vs {metric2})",
        yaxis=dict(title=metric1),
        yaxis2=dict(title=metric2, overlaying="y", side="right"),
        barmode="group"
    )

    return fig


# ---------------------------------------------------
# ✅ 2. SELECTED MODEL COMPARISON
# ---------------------------------------------------

def plot_model_comparison(df, model_list, metric=None, mode=None, task=None):

    metric = _get_metric(df, metric, task)

    if not metric:
        return px.bar(title="Metric not available")

    cleaner = DataCleaner(df)
    df = cleaner.clean([metric])

    df = df[df["model"].isin(model_list)]

    if mode:
        df = df[df["mode"] == mode]

    if df.empty:
        return px.bar(title="No valid data")

    group_col = _resolve_group_col(df)

    return px.bar(
        df,
        x=group_col,
        y=metric,
        color=group_col,
        title="Selected Model Comparison"
    )


# ---------------------------------------------------
# ✅ 3. BEST MODEL (DIRECTION-AWARE)
# ---------------------------------------------------

def plot_best_model_highlight(df, metric=None, task=None):

    metric = _get_metric(df, metric, task)

    if not metric:
        return px.bar(title="Metric not available")

    cleaner = DataCleaner(df)
    df = cleaner.clean([metric])

    direction = _get_direction(metric)

    best_idx = df[metric].idxmax() if direction == 1 else df[metric].idxmin()

    df["highlight"] = "Other"
    df.loc[best_idx, "highlight"] = "Best"

    group_col = _resolve_group_col(df)

    return px.bar(
        df,
        x=group_col,
        y=metric,
        color="highlight",
        title=f"Best Model ({metric})"
    )


# ---------------------------------------------------
# ✅ 4. BEST PER MODEL (FIXED)
# ---------------------------------------------------

def plot_best_per_model(df, metric=None, task=None):

    metric = _get_metric(df, metric, task)

    if not metric:
        return px.bar(title="Metric not available")

    cleaner = DataCleaner(df)
    df = cleaner.clean([metric])

    direction = _get_direction(metric)

    if direction == 1:
        idx = df.groupby("model")[metric].idxmax()
    else:
        idx = df.groupby("model")[metric].idxmin()

    df_best = df.loc[idx]

    return px.bar(
        df_best,
        x="model",
        y=metric,
        color="model",
        title="Best Configuration per Model"
    )


# ---------------------------------------------------
# ✅ 5. MODE COMPARISON
# ---------------------------------------------------

def plot_mode_comparison(df, metric=None, task=None):

    metric = _get_metric(df, metric, task)

    if not metric:
        return px.bar(title="Metric not available")

    cleaner = DataCleaner(df)
    df = cleaner.clean([metric])

    group_col = _resolve_group_col(df)

    return px.bar(
        df,
        x=group_col,
        y=metric,
        color="mode",
        barmode="group",
        title="Mode Comparison"
    )


# ---------------------------------------------------
# ✅ 6. PREPROCESSING IMPACT
# ---------------------------------------------------

def plot_preprocessing_impact(df, x=None, y=None, task=None):

    x = _get_metric(df, x, task)
    y = _get_metric(df, y, task)

    cleaner = DataCleaner(df)
    df = cleaner.clean([x, y])

    color_col = _resolve_group_col(df)
    symbol_col = "imputer" if "imputer" in df.columns else None

    return px.scatter(
        df,
        x=x,
        y=y,
        color=color_col,
        symbol=symbol_col,
        title="Preprocessing Impact"
    )


# ---------------------------------------------------
# ✅ 7. MODEL RANKING (FIXED)
# ---------------------------------------------------

def plot_model_ranking(df, metric=None, task=None):

    metric = _get_metric(df, metric, task)

    if not metric:
        return px.bar(title="Metric not available")

    cleaner = DataCleaner(df)
    df = cleaner.clean([metric])

    direction = _get_direction(metric)

    df = df.sort_values(metric, ascending=(direction == -1))
    df["rank"] = range(1, len(df) + 1)

    group_col = _resolve_group_col(df)

    return px.bar(
        df,
        x=group_col,
        y="rank",
        title="Model Ranking (Lower Rank = Better)",
        color=group_col
    )
