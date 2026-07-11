import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def relu(x):
    """
    ReLU (Rectified Linear Unit)

    Formula:
        ReLU(x) = max(0, x)

    Examples:
        ReLU(-2) = 0
        ReLU(0)  = 0
        ReLU(5)  = 5

    Purpose:
        Introduces non-linearity into the neural network.

    Why ReLU?
        - Simple and fast
        - Most widely used activation function
        - Helps deep networks learn complex patterns
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid Activation Function

    Formula:
        σ(x) = 1 / (1 + e^(-x))

    Output Range:
        0 to 1

    Purpose:
        Converts the final neuron output into a probability.

    Example:
        Input  = 0
        Output = 0.5
    """
    return 1 / (1 + np.exp(-x))


def main():
    """
    Forward Propagation Example

    Network Architecture
    --------------------

         Input Layer

           x1 = 1
           x2 = 2

               ↓

         Hidden Layer
           h1     h2

               ↓

         Output Layer
             y

    Objective:
        Understand how data flows from
        Input → Hidden → Output
    """

    builder = HtmlBuilder()

    # ============================================================
    # INPUT LAYER
    # ============================================================
    #
    # Input Features
    #
    # x1 = 1
    # x2 = 2
    #
    # Input Vector Shape
    # (1,2)
    #
    # X = [1, 2]
    # ============================================================

    X = np.array([1, 2])

    # ============================================================
    # HIDDEN LAYER WEIGHTS
    # ============================================================
    #
    # Each column represents a hidden neuron.
    #
    #          h1     h2
    # x1      0.2    0.4
    # x2      0.3    0.1
    #
    # Shape = (2,2)
    #
    # 2 Input Neurons
    # 2 Hidden Neurons
    # ============================================================

    W1 = np.array([
        [0.2, 0.4],
        [0.3, 0.1]
    ])

    # ============================================================
    # HIDDEN LAYER BIAS
    # ============================================================
    #
    # One bias for each hidden neuron.
    #
    # b1 = [0.1, 0.1]
    # ============================================================

    b1 = np.array([0.1, 0.1])

    # ============================================================
    # FORWARD PROPAGATION
    # INPUT → HIDDEN LAYER
    # ============================================================
    #
    # Formula:
    #
    # Hidden_Z = X.W1 + b1
    #
    # Calculation:
    #
    # Hidden_Z =
    # [1,2]
    #
    #  ×
    #
    # [[0.2,0.4],
    #  [0.3,0.1]]
    #
    # +
    #
    # [0.1,0.1]
    #
    # Result:
    #
    # Hidden_Z = [0.9, 0.7]
    # ============================================================

    hidden_z = np.dot(X, W1) + b1

    # ============================================================
    # HIDDEN LAYER ACTIVATION
    # ============================================================
    #
    # Apply ReLU to each hidden neuron.
    #
    # Hidden_A = ReLU(Hidden_Z)
    #
    # Since:
    #
    # Hidden_Z = [0.9,0.7]
    #
    # ReLU does not change positive values.
    #
    # Hidden_A = [0.9,0.7]
    # ============================================================

    hidden_a = relu(hidden_z)

    # ============================================================
    # OUTPUT LAYER WEIGHTS
    # ============================================================
    #
    # Hidden Layer (2 neurons)
    #            ↓
    # Output Layer (1 neuron)
    #
    # Shape = (2,1)
    # ============================================================

    W2 = np.array([
        [0.5],
        [0.6]
    ])

    # ============================================================
    # OUTPUT BIAS
    # ============================================================

    b2 = np.array([0.1])

    # ============================================================
    # HIDDEN → OUTPUT
    # ============================================================
    #
    # Formula:
    #
    # Output_Z =
    # Hidden_A.W2 + b2
    #
    # Calculation:
    #
    # [0.9,0.7]
    #
    # ×
    #
    # [[0.5],
    #  [0.6]]
    #
    # +
    #
    # [0.1]
    #
    # Result:
    #
    # Output_Z = 0.97
    # ============================================================

    output_z = np.dot(hidden_a, W2) + b2

    # ============================================================
    # OUTPUT ACTIVATION
    # ============================================================
    #
    # Final Prediction
    #
    # Output =
    # Sigmoid(Output_Z)
    #
    # Output =
    # Sigmoid(0.97)
    #
    # Output ≈ 0.725
    #
    # Interpretation:
    #
    # Model predicts approximately
    # 72.5% confidence for class 1.
    # ============================================================

    output = sigmoid(output_z)

    # ============================================================
    # CREATE RESULTS TABLE
    # ============================================================
    #
    # Display intermediate calculations
    # performed during forward propagation.
    # ============================================================

    results = pd.DataFrame({
        "Value": [
            hidden_z[0],
            hidden_z[1],
            hidden_a[0],
            hidden_a[1],
            output[0]
        ]
    },
        index=[
            "Hidden Z1",
            "Hidden Z2",
            "Hidden A1",
            "Hidden A2",
            "Output"
    ])

    # ============================================================
    # BUILD REPORT
    # ============================================================

    html_doc = builder.build_page(
        "Forward Propagation Report",
        builder.grid([
            builder.card(
                "Forward Propagation Results",
                builder.render_dataframe(results)
            )
        ])
    )

    # ============================================================
    # SAVE HTML REPORT
    # ============================================================

    ru.save_html_report(
        __file__,
        "forward_propagation_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
