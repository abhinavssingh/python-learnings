import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


class ConfusionMatrixPlot:
    @staticmethod
    def create_plot(
            y_true,
            y_pred,
            class_labels: list[str] | None = None,
            title: str = "Confusion Matrix",
    ):
        cm = confusion_matrix(y_true, y_pred)

        labels = class_labels or [
            f"Class {idx}"
            for idx in range(cm.shape[0])
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=460,
            width=560,
        )

        return fig

    @staticmethod
    def to_html(
            y_true,
            y_pred,
            class_labels: list[str] | None = None,
            title: str = "Confusion Matrix",
            include_plotlyjs: str | bool = False,
    ) -> str:
        fig = ConfusionMatrixPlot.create_plot(
            y_true=y_true,
            y_pred=y_pred,
            class_labels=class_labels,
            title=title,
        )

        return fig.to_html(
            full_html=False,
            include_plotlyjs=include_plotlyjs,
        )
