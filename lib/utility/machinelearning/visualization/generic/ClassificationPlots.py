import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.shared.DataCleaner import DataCleaner
from lib.utility.machinelearning.visualization.core.MetricResolver import MetricResolver


class ClassificationPlots:
    """
    Classification-specific visualizer built on generic framework.
    Supports accuracy, f1, roc_auc, precision, recall, etc.
    """

    # ----------------------------
    # FILTER
    # ----------------------------
    def _filter(self, df, model=None, mode=None, type_=None):
        df = df.copy()

        if model:
            df = df[df["model"] == model]

        if mode:
            df = df[df["mode"] == mode]

        if type_:
            df = df[df["type"] == type_]

        return df

    # ----------------------------
    # SCATTER (SMART)
    # ----------------------------
    def plot_scatter(
        self,
        df,
        x_metric=None,
        y_metric=None,
        size=None,
        color="model",
        mode=None
    ):
        df = self._filter(df, mode=mode)

        resolver = MetricResolver(df)

        # ✅ auto selection (classification-aware)
        if not x_metric or not y_metric:
            x_metric, y_metric = self._default_metrics(df)

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

    # ----------------------------
    # BAR (METRIC COMPARISON)
    # ----------------------------
    def plot_bar(self, df, metric="accuracy", group_by="model", mode=None):

        df = self._filter(df, mode=mode)

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

    # ----------------------------
    # BEST MODEL
    # ----------------------------
    def plot_best_model(self, df, metric="f1"):

        cleaner = DataCleaner(df)
        df = cleaner.clean(required_cols=[metric])

        if df.empty:
            return px.bar(title="No valid data")

        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]

        fig = px.bar(df, x="model", y=metric, color="model")

        fig.add_annotation(
            x=best_row["model"],
            y=best_row[metric],
            text=f"Best: {best_row['model']}<br>{metric}={best_row[metric]:.4f}",
            showarrow=True
        )

        return fig

    # ----------------------------
    # MULTI-METRIC COMPARISON
    # ----------------------------
    def plot_multi_metrics(self, df, metrics=None):

        if metrics is None:
            metrics = ["accuracy", "f1", "precision", "recall"]

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

    # ----------------------------
    # ROC CURVE (MULTI-MODEL ✅)
    # ----------------------------

    def plot_roc_all_models(self, results):

        import numpy as np
        import plotly.graph_objects as go

        fig = go.Figure()

        best_model = None
        best_auc = -1

        for r in results:

            model_name = r["model"]
            auc = r.get("roc_auc")
            roc_data = r.get("artifacts", {}).get("roc_curve")

            if not roc_data:
                continue

            fpr = np.array(roc_data.get("fpr"))
            tpr = np.array(roc_data.get("tpr"))
            thresholds = roc_data.get("thresholds")

            if fpr is None or tpr is None:
                continue

            # ✅ find best threshold (Youden’s J)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)

            best_fpr = fpr[best_idx]
            best_tpr = tpr[best_idx]

            best_thr = None
            if thresholds is not None and len(thresholds) > best_idx:
                best_thr = thresholds[best_idx]

            # ✅ track best model
            if auc and auc > best_auc:
                best_auc = auc
                best_model = model_name

            # ✅ plot ROC line
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={auc:.3f})" if auc else model_name,
                line=dict(width=4 if model_name == best_model else 2)
            ))

            # ✅ add threshold marker
            fig.add_trace(go.Scatter(
                x=[best_fpr],
                y=[best_tpr],
                mode="markers+text",
                text=[f"{model_name}<br>thr={best_thr:.2f}" if best_thr else model_name],
                textposition="top center",
                marker=dict(size=10, color="red"),
                showlegend=False
            ))

        # ✅ diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random"
        ))

        # ✅ best model annotation
        if best_model:
            fig.add_annotation(
                text=f"🏆 Best Model: {best_model} (AUC={best_auc:.3f})",
                x=0.6,
                y=0.2,
                showarrow=False,
                font=dict(size=14)
            )

        fig.update_layout(
            title="ROC Curve Comparison (Best Threshold Highlighted)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        return fig
    # ----------------------------
    # AUTO METRIC PICK (IMPORTANT)
    # ----------------------------

    def _default_metrics(self, df):

        # ✅ classification priority
        if "roc_auc" in df.columns:
            return "roc_auc", "f1"

        if "f1" in df.columns:
            return "f1", "accuracy"

        return "accuracy", "precision"

    # ----------------------------
    # AUTO PLOT
    # ----------------------------
    def auto_plot(self, df):

        x_metric, y_metric = self._default_metrics(df)

        return self.plot_scatter(df, x_metric, y_metric)
