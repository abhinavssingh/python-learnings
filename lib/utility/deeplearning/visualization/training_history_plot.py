import plotly.graph_objects as go


class TrainingHistoryPlot:

    @staticmethod
    def create_plot(history):

        fig = go.Figure()

        # Loss
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history.history["loss"]) + 1)),
                y=history.history["loss"],
                mode="lines+markers",
                name="Training Loss"
            )
        )

        if "val_loss" in history.history:

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history.history["val_loss"]) + 1)),
                    y=history.history["val_loss"],
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

        fig = go.Figure()

        if "accuracy" in history.history:

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history.history["accuracy"]) + 1)),
                    y=history.history["accuracy"],
                    mode="lines+markers",
                    name="Training Accuracy"
                )
            )

        if "val_accuracy" in history.history:

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history.history["val_accuracy"]) + 1)),
                    y=history.history["val_accuracy"],
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
