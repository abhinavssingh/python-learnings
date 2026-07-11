import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.reports.report_utils import ReportUtils as ru


class SinglePerceptron:
    """
    A simple implementation of a single perceptron.

    This perceptron is configured to behave like an AND gate:

        x1  x2  Output
        0   0     0
        0   1     0
        1   0     0
        1   1     1

    Formula:

        z = (x1 * w1) + (x2 * w2) + bias

    Output:

        y = step(z)

    If z >= 0 -> 1
    Else       -> 0
    """

    def __init__(self, df):
        self.df = df
        # Weight associated with each input neuron.
        np.random.seed(42)
        self.weights = np.array([0.2, -0.4])

        # Bias shifts the decision boundary.
        self.bias = 0.3

    def train(self, X, y, epochs=10, learning_rate=0.1):
        """
        Train the perceptron using the Perceptron Learning Algorithm.

        Training Process
        ----------------
        1. Take an input sample.
        2. Calculate prediction.
        3. Compare prediction with actual value.
        4. Compute error.

            error = actual - predicted

        5. Update weights and bias.
        6. Repeat for all samples and epochs.

        Parameters
        ----------
        X : ndarray
            Training feature matrix.

        y : ndarray
            Target labels.

        epochs : int
            Number of complete passes through dataset.

        learning_rate : float
            Controls update magnitude.

        Returns
        -------
        list
            History of weights and bias values used for animation.
        """

        history = []

        for epoch in range(epochs):

            for inputs, target in zip(X, y):

                prediction = self.predict(inputs)

                error = target - prediction

                # ------------------------------------------------------------------
                # Perceptron Learning Rule
                #
                # New Weight =
                #     Old Weight + Learning Rate × Error × Input
                #
                # If prediction is wrong:
                #     weights move toward the correct answer.
                #
                # If prediction is correct:
                #     error becomes 0 and weights remain unchanged.
                # ------------------------------------------------------------------

                self.weights += learning_rate * error * np.array(inputs)

                # Bias is updated exactly like a weight connected to a constant input.
                self.bias += learning_rate * error

            history.append({
                "epoch": epoch + 1,
                "weights": self.weights.copy(),
                "bias": self.bias
            })

        return history

    def step_function(self, z):
        """
        Activation Function

        Converts the weighted sum into a binary output.

        Example:
            z = 0.5  -> 1
            z = -0.2 -> 0
        """
        return 1 if z >= 0 else 0

    def predict(self, inputs):
        """
        Perform Forward Propagation.

        A perceptron makes predictions in two stages.

        Stage 1: Weighted Sum

            z = x1*w1 + x2*w2 + b

        Stage 2: Activation Function

            output = step(z)

        If z >= 0:
            output = 1

        Else:
            output = 0

        Returns
        -------
        int
            Predicted class label and all intermediate values.
        """

        # Dot product:
        # (x1*w1 + x2*w2)
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Apply activation function
        return self.step_function(weighted_sum)

    def calculate(self, inputs):
        """
        Perform forward propagation and return all intermediate values.
        """

        x1 = inputs[0]
        x2 = inputs[1]

        w1 = self.weights[0]
        w2 = self.weights[1]

        z = (x1 * w1) + (x2 * w2) + self.bias

        output = self.step_function(z)

        return {
            "x1": x1,
            "x2": x2,
            "w1": w1,
            "w2": w2,
            "bias": self.bias,
            "z": z,
            "output": output
        }

    def create_decision_boundary_plot(self):
        """
        Visualize how the perceptron separates classes.

            Decision Boundary:

                w1*x1 + w2*x2 + b = 0

            For current weights:

                x1 + x2 - 1.5 = 0

            Region Above Boundary  -> Prediction = 1
            Region Below Boundary  -> Prediction = 0
            """

        # Generate grid points
        x = np.linspace(-0.2, 1.2, 100)
        y = np.linspace(-0.2, 1.2, 100)

        xx, yy = np.meshgrid(x, y)

        # Compute perceptron predictions for every point
        z = np.array([
            self.predict([x1, x2])
            for x1, x2 in zip(xx.ravel(), yy.ravel())
        ])

        z = z.reshape(xx.shape)

        fig = go.Figure()

        # ------------------------------------------------------------------
        # Decision regions
        # ------------------------------------------------------------------
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=z,
                colorscale=[
                    [0, "#ffcccc"],
                    [0.5, "#ffcccc"],
                    [0.5, "#ccffcc"],
                    [1, "#ccffcc"]
                ],
                showscale=False,
                opacity=0.5,
                hoverinfo="skip",
                contours=dict(
                    start=0,
                    end=1,
                    size=1
                ),
                name="Decision Regions"
            )
        )

        # ------------------------------------------------------------------
        # Decision boundary
        # ------------------------------------------------------------------
        boundary_x = np.linspace(-0.2, 1.2, 100)
        boundary_y = 1.5 - boundary_x

        fig.add_trace(
            go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode="lines",
                name="Decision Boundary",
                line=dict(
                    color="blue",
                    width=4
                )
            )
        )

        # ------------------------------------------------------------------
        # Training points
        # ------------------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=[0, 0, 1],
                y=[0, 1, 0],
                mode="markers+text",
                name="Class 0",
                marker=dict(
                    size=15,
                    color="red"
                ),
                text=["(0,0)", "(0,1)", "(1,0)"],
                textposition="top center"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[1],
                y=[1],
                mode="markers+text",
                name="Class 1",
                marker=dict(
                    size=18,
                    color="green"
                ),
                text=["(1,1)"],
                textposition="top center"
            )
        )

        fig.update_layout(
            title="Single Perceptron Decision Boundary (AND Gate)",
            xaxis_title="Input x1",
            yaxis_title="Input x2",
            template="plotly_white",
            height=450,
            width=700,
            xaxis=dict(range=[-0.2, 1.2]),
            yaxis=dict(range=[-0.2, 1.2])
        )

        return fig

    def create_training_animation(self, history):
        """
        Animate decision boundary movement during training.
        """

        frames = []

        for step in history:

            w1 = step["weights"][0]
            w2 = step["weights"][1]
            b = step["bias"]

            x_line = np.linspace(-0.2, 1.2, 100)

            if abs(w2) > 1e-6:
                y_line = -(w1 * x_line + b) / w2
            else:
                y_line = np.zeros_like(x_line)

            frame = go.Frame(
                name=f"Epoch {step['epoch']}",
                data=[
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="blue", width=4)
                    )
                ],
                traces=[2]   # Update trace index 2 (Decision Boundary)
            )

            frames.append(frame)

        # Initial boundary
        first = history[0]

        x_line = np.linspace(-0.2, 1.2, 100)

        y_line = -(first["weights"][0] * x_line + first["bias"]) / first["weights"][1]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=self.df[self.df["Actual"] == 0]["x1"],
                    y=self.df[self.df["Actual"] == 0]["x2"],
                    mode="markers+text",
                    name="Class 0",
                    marker=dict(size=15, color="red"),
                ),
                go.Scatter(
                    x=self.df[self.df["Actual"] == 1]["x1"],
                    y=self.df[self.df["Actual"] == 1]["x2"],
                    mode="markers+text",
                    name="Class 1",
                    marker=dict(size=18, color="green"),
                ),
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Decision Boundary",
                    line=dict(color="blue", width=4)
                )
            ],
            frames=frames
        )

        fig.update_layout(
            title="Perceptron Training Animation",
            template="plotly_white",
            height=450,
            xaxis=dict(range=[-0.2, 1.2]),
            yaxis=dict(range=[-0.2, 1.2]),
            xaxis_title="x1",
            yaxis_title="x2",
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": 500,
                                        "redraw": True
                                    },
                                    "transition": {
                                        "duration": 200
                                    },
                                    "fromcurrent": True
                                }
                            ]
                        }
                    ]
                }
            ]
        )

        return fig


def main():
    """
    Demonstration of a single perceptron solving an AND gate.

    Steps:
    1. Create perceptron
    2. Create AND gate dataset
    3. Generate predictions
    4. Display results in an HTML report
    """

    builder = HtmlBuilder()
    plot_renderer = PlotRenderer()
    content = []

    # AND Gate Training Dataset
    #
    # x1  x2  Actual
    # 0   0     0
    # 0   1     0
    # 1   0     0
    # 1   1     1
    #
    data = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]

    # Convert dataset into DataFrame for better visualization
    df = pd.DataFrame(data, columns=["x1", "x2", "Actual"])

    # Create perceptron instance
    perceptron = SinglePerceptron(df)

    # Run prediction for every row in the dataset
    df["Predicted"] = df.apply(
        lambda row: perceptron.predict(
            [row["x1"], row["x2"]]
        ),
        axis=1
    )

    X = df[["x1", "x2"]].values
    y = df["Actual"].values

    history = perceptron.train(X, y, epochs=15, learning_rate=0.2)

    print("\nTraining History")

    for h in history:
        print(
            f"Epoch={h['epoch']} "
            f"Weights={h['weights']} "
            f"Bias={h['bias']}"
        )

    calculation_rows = []

    for _, row in df.iterrows():

        result = perceptron.calculate(
            [row["x1"], row["x2"]]
        )

        calculation_rows.append({

            "x1": result["x1"],
            "x2": result["x2"],

            "w1": round(result["w1"], 2),
            "w2": round(result["w2"], 2),

            "Bias": round(result["bias"], 2),

            "Weighted Sum Formula":
                f"({result['x1']}×{result['w1']:.1f}) + "
                f"({result['x2']}×{result['w2']:.1f}) + "
                f"({result['bias']:.1f})",

            "z": round(result["z"], 2),

            "Activation":
                "step(z)",

            "Predicted":
                result["output"],

            "Actual":
                row["Actual"]
        })

    calculation_df = pd.DataFrame(calculation_rows)

    architecture_pre = """
    Input Layer

    x1 ----\
            \
            > Weighted Sum ---> Activation ---> Output
            /
    x2 ----/

    z = x1*w1 + x2*w2 + b

    output = step(z)
    """

    # Build HTML report
    content.append(builder.grid([
        builder.card("AND Gate Dataset", builder.render_dataframe(df)),
        builder.card("Perceptron Parameters", builder.render_dict({
            "Weight 1": perceptron.weights[0],
            "Weight 2": perceptron.weights[1],
            "Bias": perceptron.bias
        })),
        builder.card("Perceptron Architecture", builder.render_pre(architecture_pre)),
        builder.card("Forward Propagation Calculations", builder.render_dataframe(calculation_df))
    ]))

    # create decision boundary plot
    animation_fig = perceptron.create_training_animation(history)

    decision_boundary_fig = perceptron.create_decision_boundary_plot()

    content.append(builder.chart_grid([
        plot_renderer.plot_to_card(decision_boundary_fig, "Decision Boundary Visualization"),
        plot_renderer.plot_to_card(animation_fig, "Training Animation")
    ]))

    html_doc = builder.build_page(
        "Single Perceptron Report",
        "\n".join(content)
    )

    # Save report to:
    # reports/single_perceptron_report.html
    ru.save_html_report(
        __file__,
        "single_perceptron_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
