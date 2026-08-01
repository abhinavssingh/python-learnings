import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve


class ROCurvePlot:
    @staticmethod
    def create_plot(
        y_true,
        y_prob,
        title: str = "ROC Curve",
    ):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC={auc_score:.4f})",
                line=dict(color="#1F77B4", width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Baseline",
                line=dict(color="#7F8C8D", dash="dash"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420,
        )

        return fig, float(auc_score)

    @staticmethod
    def to_html(
        y_true,
        y_prob,
        title: str = "ROC Curve",
        include_plotlyjs: str | bool = False,
    ) -> tuple[str, float]:
        fig, auc_score = ROCurvePlot.create_plot(
            y_true=y_true,
            y_prob=y_prob,
            title=title,
        )

        return (
            fig.to_html(
                full_html=False,
                include_plotlyjs=include_plotlyjs,
            ),
            auc_score,
        )
