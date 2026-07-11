import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def sigmoid(x):
    """
    Sigmoid Activation Function

    Formula:
        σ(x) = 1 / (1 + e^(-x))

    Output Range:
        0 to 1

    Purpose:
        Introduces non-linearity and converts neuron outputs
        into probability-like values.

    Example:
        Input  = 0
        Output = 0.5
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of Sigmoid Function

    Formula:
        σ'(x) = σ(x) * (1 - σ(x))

    Purpose:
        Used during backpropagation to calculate gradients
        and determine how much each weight should change.

    Example:
        x = 0.5

        derivative =
            0.5 * (1 - 0.5)

        derivative = 0.25
    """
    return x * (1 - x)


def main():
    """
    Multi Layer Perceptron (MLP) Learning Example

    Network Architecture
    --------------------

    Input Layer
        x1
        x2

          ↓

    Hidden Layer
        h1 h2 h3 h4

          ↓

    Output Layer
        y

    Task:
        Learn XOR Logic

        x1  x2  Output
        0   0     0
        0   1     1
        1   0     1
        1   1     0

    XOR cannot be solved by a Single Perceptron.
    It requires at least one hidden layer.
    """

    builder = HtmlBuilder()

    # ============================================================
    # Training Dataset (XOR Gate)
    # ============================================================
    #
    # Input Features
    #
    # x1 x2
    #
    # Target Output
    #
    # y
    # ============================================================

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # ------------------------------------------------------------
    # Fix random seed so generated weights remain the same
    # every time we run the script.
    # ------------------------------------------------------------
    np.random.seed(42)

    # ============================================================
    # Weight Initialization
    # ============================================================
    #
    # W1
    #
    # Input Layer (2 neurons)
    #            →
    # Hidden Layer (4 neurons)
    #
    # Shape = (2,4)
    #
    # W2
    #
    # Hidden Layer (4 neurons)
    #            →
    # Output Layer (1 neuron)
    #
    # Shape = (4,1)
    # ============================================================

    W1 = np.random.randn(2, 4)
    W2 = np.random.randn(4, 1)

    # Store loss values for every epoch
    losses = []

    # ============================================================
    # Training Loop
    # ============================================================

    for epoch in range(1000):

        # ========================================================
        # FORWARD PROPAGATION
        # ========================================================
        #
        # Step 1
        #
        # Hidden Layer Calculation
        #
        # hidden_z = X · W1
        #
        # hidden_a = sigmoid(hidden_z)
        # ========================================================

        hidden = sigmoid(np.dot(X, W1))

        # ========================================================
        # Step 2
        #
        # Output Layer Calculation
        #
        # output_z = hidden · W2
        #
        # output = sigmoid(output_z)
        # ========================================================

        output = sigmoid(np.dot(hidden, W2))

        # ========================================================
        # ERROR CALCULATION
        # ========================================================
        #
        # Error =
        # Actual - Predicted
        #
        # Larger error means the network prediction
        # is further away from the expected value.
        # ========================================================

        error = y - output

        # ========================================================
        # LOSS FUNCTION
        # ========================================================
        #
        # Mean Squared Error (MSE)
        #
        # Loss = Mean((Actual - Predicted)^2)
        #
        # Goal:
        # Minimize loss over time.
        # ========================================================

        losses.append([
            epoch + 1,
            np.mean(np.square(error))
        ])

        # ========================================================
        # BACKPROPAGATION
        # ========================================================
        #
        # Calculate gradient at output layer.
        #
        # delta =
        # error × derivative(sigmoid)
        #
        # Determines how much the output layer
        # contributed to the total error.
        # ========================================================

        output_delta = error * sigmoid_derivative(output)

        # ========================================================
        # Propagate error backwards
        #
        # Hidden Error =
        # Output Delta × W2^T
        #
        # This tells hidden neurons how much
        # they contributed to the final error.
        # ========================================================

        hidden_error = output_delta.dot(W2.T)

        # ========================================================
        # Hidden Layer Gradients
        # ========================================================

        hidden_delta = hidden_error * sigmoid_derivative(hidden)

        # ========================================================
        # GRADIENT DESCENT
        # ========================================================
        #
        # Weight Update Rule
        #
        # New Weight =
        # Old Weight +
        # Learning Rate × Gradient
        #
        # Learning Rate = 0.1
        # ========================================================

        W2 += hidden.T.dot(output_delta) * 0.1

        W1 += X.T.dot(hidden_delta) * 0.1

    # ============================================================
    # Predictions after training completes
    # ============================================================

    prediction_df = pd.DataFrame({
        "Actual": y.flatten(),
        "Predicted Probability": np.round(output.flatten(), 3),
        "Predicted Class": np.round(output.flatten()).astype(int)
    })

    # ============================================================
    # Loss History
    # ============================================================

    loss_df = pd.DataFrame(
        losses,
        columns=["Epoch", "Loss"]
    )

    # ============================================================
    # Build HTML Report
    # ============================================================

    html_doc = builder.build_page(
        "Multi Layer Perceptron Report",
        builder.grid([
            builder.card(
                "XOR Predictions",
                builder.render_dataframe(prediction_df)
            ),
            builder.card(
                "Training History (Last 20 Epochs)",
                builder.render_dataframe(
                    loss_df.tail(20)
                )
            )
        ])
    )

    ru.save_html_report(
        __file__,
        "multi_layer_perceptron_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
