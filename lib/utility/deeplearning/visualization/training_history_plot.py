import plotly.graph_objects as go


class TrainingHistoryPlot:

    @staticmethod
    def _to_history_dict(history):
        if hasattr(history, "history"):
            return history.history

        if isinstance(history, dict):
            return history

        raise TypeError("Unsupported history type")

    @staticmethod
    def create_plot(history):

        history_dict = TrainingHistoryPlot._to_history_dict(history)

        fig = go.Figure()

        # Loss
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history_dict["loss"]) + 1)),
                y=history_dict["loss"],
                mode="lines+markers",
                name="Training Loss"
            )
        )

        if "val_loss" in history_dict:

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history_dict["val_loss"]) + 1)),
                    y=history_dict["val_loss"],
                    mode="lines+markers",
                    name="Validation Loss"
                )
            )

        fig.update_layout(
            title="Loss Curve",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=500
        )

        return fig

    @staticmethod
    def create_accuracy_plot(history):

        history_dict = TrainingHistoryPlot._to_history_dict(history)

        fig = go.Figure()

        if "accuracy" in history_dict:

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history_dict["accuracy"]) + 1)),
                    y=history_dict["accuracy"],
                    mode="lines+markers",
                    name="Training Accuracy"
                )
            )

        if "val_accuracy" in history_dict:

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history_dict["val_accuracy"]) + 1)),
                    y=history_dict["val_accuracy"],
                    mode="lines+markers",
                    name="Validation Accuracy"
                )
            )

        fig.update_layout(
            title="Accuracy Curve",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            height=500
        )

        return fig

    @staticmethod
    def create_metric_plot(
        history,
        train_metric: str,
        val_metric: str | None = None,
        title: str | None = None,
        yaxis_title: str | None = None,
    ):
        history_dict = TrainingHistoryPlot._to_history_dict(history)

        if train_metric not in history_dict:
            raise KeyError(f"Metric '{train_metric}' not found in history")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history_dict[train_metric]) + 1)),
                y=history_dict[train_metric],
                mode="lines+markers",
                name=f"Train {train_metric}",
            )
        )

        if val_metric and val_metric in history_dict:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history_dict[val_metric]) + 1)),
                    y=history_dict[val_metric],
                    mode="lines+markers",
                    name=f"Validation {val_metric}",
                )
            )

        fig.update_layout(
            title=title or f"{train_metric} Curve",
            xaxis_title="Epoch",
            yaxis_title=yaxis_title or train_metric,
            height=500,
        )

        return fig
