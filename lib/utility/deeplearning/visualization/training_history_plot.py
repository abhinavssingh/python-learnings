import plotly.graph_objects as go


class TrainingHistoryPlot:

    @staticmethod
    def plot(history):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=history.history["loss"],
                name="Training Loss"
            )
        )

        if "val_loss" in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history["val_loss"],
                    name="Validation Loss"
                )
            )

        fig.update_layout(
            title="Training History",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )

        fig.show()
