import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.evaluation.MetricResolver import MetricResolver
from lib.utility.machinelearning.shared.DataCleaner import DataCleaner

# ---------------------------------------------------
# ✅ INTERNAL HELPERS
# ---------------------------------------------------


def _filter(df, model=None, mode=None, type_=None):

    df = df.copy()

    if model:
        df = df[df["model"] == model]

    if mode and "mode" in df.columns:
        df = df[df["mode"] == mode]

    if type_ and "type" in df.columns:
        df = df[df["type"] == type_]

    return df


def _get_metrics(df, x_metric=None, y_metric=None):

    task = "classification"

    if not x_metric or not y_metric:
        metrics = MetricResolver.get_default_metrics(task)

        if len(metrics) >= 2:
            return metrics[0], metrics[1]

    return x_metric, y_metric


# ---------------------------------------------------
# ✅ SCATTER (SMART)
# ---------------------------------------------------

def plot_scatter(df, x_metric=None, y_metric=None, size=None, color="model", mode=None):

    df = _filter(df, mode=mode)

    x_metric, y_metric = _get_metrics(df, x_metric, y_metric)

    if x_metric not in df.columns or y_metric not in df.columns:
        return px.scatter(title="Metrics not found")

    cleaner = DataCleaner(df)
    df = cleaner.clean(required_cols=[x_metric, y_metric])

    if df.empty:
        return px.scatter(title="No valid data")

    if size and size in df.columns:
        df[size] = df[size].fillna(df[size].median())

    return px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        size=size,
        color=color,
        title=f"{y_metric} vs {x_metric}"
    )


# ---------------------------------------------------
# ✅ BAR COMPARISON
# ---------------------------------------------------

def plot_bar(df, metric=None, group_by="model", mode=None):

    metric = metric or MetricResolver.get_best_metric("classification")

    df = _filter(df, mode=mode)

    cleaner = DataCleaner(df)
    df = cleaner.clean(required_cols=[metric])

    if df.empty:
        return px.bar(title="No valid data")

    return px.bar(
        df,
        x=group_by,
        y=metric,
        color=group_by,
        title=f"{metric} Comparison"
    )


# ---------------------------------------------------
# ✅ BEST MODEL (DIRECTION AWARE)
# ---------------------------------------------------

def plot_best_model(df, metric=None):

    metric = metric or MetricResolver.get_best_metric("classification")

    cleaner = DataCleaner(df)
    df = cleaner.clean(required_cols=[metric])

    if df.empty:
        return px.bar(title="No valid data")

    direction = MetricResolver.get_direction(metric)

    best_idx = df[metric].idxmax() if direction == 1 else df[metric].idxmin()
    best_row = df.loc[best_idx]

    fig = px.bar(df, x="model", y=metric, color="model")

    fig.add_annotation(
        x=best_row["model"],
        y=best_row[metric],
        text=f"Best: {best_row['model']}<br>{metric}={best_row[metric]:.4f}",
        showarrow=True
    )

    return fig


# ---------------------------------------------------
# ✅ MULTI METRIC COMPARISON
# ---------------------------------------------------

def plot_multi_metrics(df, metrics=None):

    if metrics is None:
        metrics = [
            "accuracy",
            "f1_weighted",
            "f1_macro",
            "roc_auc"
        ]

    cleaner = DataCleaner(df)
    df = cleaner.clean(required_cols=metrics)

    if df.empty:
        return px.bar(title="No valid data")

    df_melt = df.melt(
        id_vars=["model"],
        value_vars=metrics,
        var_name="metric",
        value_name="score"
    )

    return px.bar(
        df_melt,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Multi-Metric Comparison"
    )


# ---------------------------------------------------
# ✅ ROC CURVE (WITH THRESHOLD SUPPORT)
# ---------------------------------------------------

def plot_roc_all_models(artifacts):

    fig = go.Figure()

    best_model = None
    best_auc = -1

    for r in artifacts:

        model_name = r["model"]
        auc_val = r.get("roc_auc")
        roc_data = r.get("roc_curve")

        if not roc_data:
            continue

        fpr = roc_data.get("fpr")
        tpr = roc_data.get("tpr")
        thresholds = roc_data.get("thresholds")

        # ✅ Track best model
        if auc_val is not None and auc_val > best_auc:
            best_auc = auc_val
            best_model = model_name

        # =========================================================
        # ✅ CASE 1: BINARY
        # =========================================================
        if isinstance(fpr, (list, np.ndarray)):

            fpr = np.array(fpr)
            tpr = np.array(tpr)

            if len(fpr) == 0:
                continue

            # ✅ thresholds
            if thresholds is not None:
                thresholds = np.array(thresholds)

                if len(thresholds) != len(fpr):
                    thresholds = np.resize(thresholds, len(fpr))

                threshold_text = [f"T={t:.3f}" for t in thresholds]
            else:
                threshold_text = None

            # ✅ ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={auc_val:.3f})",
                line=dict(width=4 if model_name == best_model else 2),
                text=threshold_text,
                hovertemplate=(
                    "FPR: %{x:.3f}<br>"
                    "TPR: %{y:.3f}<br>"
                    "%{text}<extra></extra>"
                ) if threshold_text else None
            ))

            # ✅ best threshold (binary only)
            best_t = r.get("best_threshold")
            if best_t is not None and thresholds is not None:
                idx = np.argmin(np.abs(thresholds - best_t))

                fig.add_trace(go.Scatter(
                    x=[fpr[idx]],
                    y=[tpr[idx]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name=f"{model_name} Best T={best_t:.2f}",
                    showlegend=False
                ))

        # =========================================================
        # ✅ CASE 2: MULTICLASS (MACRO ROC)
        # =========================================================
        elif isinstance(fpr, dict):

            try:
                all_fpr = np.unique(
                    np.concatenate([np.array(fpr[k]) for k in fpr])
                )

                mean_tpr = np.zeros_like(all_fpr)

                for k in fpr:
                    mean_tpr += np.interp(
                        all_fpr,
                        np.array(fpr[k]),
                        np.array(tpr[k])
                    )

                mean_tpr /= len(fpr)

                fig.add_trace(go.Scatter(
                    x=all_fpr,
                    y=mean_tpr,
                    mode="lines",
                    name=f"{model_name} (macro AUC={auc_val:.3f})",
                    line=dict(
                        width=4 if model_name == best_model else 2,
                        dash="dash"
                    )
                ))

            except Exception as e:
                print(f"Multiclass ROC failed for {model_name}: {e}")

    # ✅ Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash"),
        name="Random"
    ))

    fig.update_layout(
        title="ROC Curve (Binary + Multiclass)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    return fig

# ---------------------------------------------------
# ✅ PR CURVE (WITH THRESHOLD SUPPORT)
# ---------------------------------------------------


def plot_pr_all_models(artifacts):

    fig = go.Figure()

    best_model = None
    best_auc = -1

    for r in artifacts:

        model_name = r["model"]
        pr_auc_val = r.get("pr_auc")
        pr_data = r.get("pr_curve")

        if not pr_data:
            continue

        precision = pr_data.get("precision")
        recall = pr_data.get("recall")
        thresholds = pr_data.get("thresholds")

        # ✅ Track best model
        if pr_auc_val is not None and pr_auc_val > best_auc:
            best_auc = pr_auc_val
            best_model = model_name

        # =========================================================
        # ✅ CASE 1: BINARY
        # =========================================================
        if isinstance(precision, (list, np.ndarray)):

            precision = np.array(precision)
            recall = np.array(recall)

            if len(precision) == 0:
                continue

            # ✅ Threshold alignment
            if thresholds is not None:
                thresholds = np.array(thresholds)

                if len(thresholds) == len(precision) - 1:
                    thresholds = np.append(thresholds, 1.0)

                threshold_text = [f"T={t:.3f}" for t in thresholds]
            else:
                threshold_text = None

            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode="lines+markers",
                name=f"{model_name} (PR-AUC={pr_auc_val:.3f})" if pr_auc_val else model_name,
                line=dict(width=4 if model_name == best_model else 2),
                text=threshold_text,
                hovertemplate=(
                    "Recall: %{x:.3f}<br>"
                    "Precision: %{y:.3f}<br>"
                    "%{text}<extra></extra>"
                ) if threshold_text else None
            ))

        # =========================================================
        # ✅ CASE 2: MULTICLASS (MACRO PR)
        # =========================================================
        elif isinstance(precision, dict):

            try:
                all_recall = np.unique(
                    np.concatenate([np.array(recall[k]) for k in recall])
                )

                mean_precision = np.zeros_like(all_recall)

                for k in precision:
                    mean_precision += np.interp(
                        all_recall,
                        np.array(recall[k]),
                        np.array(precision[k])
                    )

                mean_precision /= len(precision)

                fig.add_trace(go.Scatter(
                    x=all_recall,
                    y=mean_precision,
                    mode="lines",
                    name=(
                        f"{model_name} (macro PR-AUC={pr_auc_val:.3f})"
                        if pr_auc_val is not None else model_name
                    ),
                    line=dict(
                        width=4 if model_name == best_model else 2,
                        dash="dash"
                    )
                ))

            except Exception as e:
                print(f"Multiclass PR failed for {model_name}: {e}")

    fig.update_layout(
        title="Precision-Recall Curve (Binary + Multiclass)",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    return fig
# ---------------------------------------------------
# ✅ AUTO ENTRY (FOR ENGINE)
# ---------------------------------------------------


def plot_all(results, artifacts=None):

    df = pd.DataFrame(results)

    return {
        "scatter": plot_scatter(df),
        "bar": plot_bar(df),
        "multi_metric": plot_multi_metrics(df),
        "best_model": plot_best_model(df),
        "roc_curve": plot_roc_all_models(artifacts),
        "pr_curve": plot_pr_all_models(artifacts)
    }
