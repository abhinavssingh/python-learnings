import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def step_function(x):
    """
    Step Activation Function

    Formula:

        Step(x) = 1  if x >= 0
                  0  otherwise

    Output Range:
        {0, 1}

    Purpose:
        Converts a continuous value into a binary decision.

    Example:

        Input = -2  → Output = 0
        Input =  5  → Output = 1

    Used In:
        Single Perceptron
    """

    return np.where(x >= 0, 1, 0)


def sigmoid(x):
    """
    Sigmoid Activation Function

    Formula:

        σ(x) = 1 / (1 + e^(-x))

    Output Range:
        0 to 1

    Purpose:
        Converts any input value into a probability.

    Example:

        Input = 0
        Output = 0.5

    Why Important?
        Commonly used in binary classification problems.

    Used In:
        Logistic Regression
        Neural Networks
    """

    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Tanh (Hyperbolic Tangent)

    Formula:

        tanh(x)

    Output Range:
        -1 to 1

    Purpose:
        Similar to Sigmoid but centered around zero.

    Example:

        Input = -5 → ≈ -1
        Input =  0 →  0
        Input =  5 → ≈ 1

    Benefit:
        Zero-centered outputs help optimization.
    """

    return np.tanh(x)


def relu(x):
    """
    ReLU (Rectified Linear Unit)

    Formula:

        ReLU(x) = max(0, x)

    Output Range:
        0 to +∞

    Example:

        Input = -5 → 0
        Input =  0 → 0
        Input =  5 → 5

    Why Important?

        - Fast computation
        - Prevents some vanishing gradient problems
        - Most widely used activation function in Deep Learning

    Used In:
        Hidden Layers of Neural Networks
        CNNs
        Deep Neural Networks
    """

    return np.maximum(0, x)


def main():
    """
    Activation Functions Learning Example

    Objective
    ---------

    Understand how different activation functions
    transform the same input values.

    Input Values:

        -5, -4, -3, ..., 4, 5

    Activation Functions:

        1. Step Function
        2. Sigmoid
        3. Tanh
        4. ReLU

    Why Activation Functions?

        Without activation functions,
        neural networks behave like
        simple linear equations.

        Activation functions introduce
        non-linearity which enables neural
        networks to learn complex patterns.
    """

    builder = HtmlBuilder()

    # ============================================================
    # Generate input values from -5 to 5
    #
    # These values are passed through each activation function
    # to compare how each function behaves.
    # ============================================================

    x = np.arange(-5, 6, 1)

    # ============================================================
    # Apply all activation functions
    #
    # Every row shows how a particular input
    # is transformed by different activations.
    # ============================================================

    df = pd.DataFrame({
        "Input": x,
        "Step": step_function(x),
        "Sigmoid": np.round(sigmoid(x), 4),
        "Tanh": np.round(tanh(x), 4),
        "ReLU": np.round(relu(x), 4)
    })

    comparison_df = pd.DataFrame({
        "Activation": ["Step", "Sigmoid", "Tanh", "ReLU"],
        "Output Range": ["0-1", "0-1", "-1 to 1", "0 to ∞"],
        "Differentiable": ["No", "Yes", "Yes", "Yes"],
        "Used in Deep Learning": ["No", "Sometimes", "Sometimes", "Yes"]
    })

    # ============================================================
    # Create learning notes explaining the behavior
    # of each activation function.
    # ============================================================

    html_doc = builder.build_page(
        "Activation Functions Fundamentals",
        builder.grid([
            builder.card("Input Values", builder.render_dataframe(pd.DataFrame({"Input": x}))),
            builder.card("Activation Function Results", builder.render_dataframe(df)),
            builder.card("Activation Function Comparison", builder.render_dataframe(comparison_df)),
            builder.card("Learning Notes", builder.render_pre("""
STEP FUNCTION
-------------
Output:
    0 or 1

Use Case:
    Single Perceptron

Problem:
    Not differentiable for backpropagation.


SIGMOID
-------
Output Range:
    0 to 1

Use Case:
    Binary Classification

Interpretation:
    Probability


TANH
----
Output Range:
    -1 to 1

Benefit:
    Zero-centered output


RELU
----
Output Range:
    0 to Positive Infinity

Benefits:
    Fast computation
    Most common activation function
    Used in modern deep learning
                """)
                         )
        ])
    )

    # ============================================================
    # Save HTML Report
    #
    # Report Location:
    # reports/activation_functions_report.html
    # ============================================================

    ru.save_html_report(
        __file__,
        "activation_functions_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
