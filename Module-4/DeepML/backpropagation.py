import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    """
    Backpropagation Example

    Objective
    ---------

    Understand how a neural network learns from mistakes.

    Backpropagation consists of:

        1. Generate Prediction
        2. Calculate Error
        3. Calculate Loss
        4. Calculate Gradient
        5. Update Weight

    Goal:

        Reduce future prediction error by adjusting weights.
    """

    builder = HtmlBuilder()

    # ============================================================
    # ACTUAL OUTPUT
    # ============================================================
    #
    # Expected value from the dataset.
    #
    # Example:
    #
    # Actual = 1
    #
    # Means:
    # The desired prediction is 1.
    # ============================================================

    actual = 1

    # ============================================================
    # MODEL PREDICTION
    # ============================================================
    #
    # Output produced by the neural network.
    #
    # Example:
    #
    # Predicted = 0.8
    #
    # This means the network predicts 80%
    # confidence for class 1.
    # ============================================================

    predicted = 0.8

    # ============================================================
    # CURRENT WEIGHT
    # ============================================================
    #
    # Weight controls the influence of an input
    # on the final prediction.
    #
    # Larger weight = greater influence.
    # ============================================================

    weight = 0.5

    # ============================================================
    # LEARNING RATE
    # ============================================================
    #
    # Controls how large each update should be.
    #
    # Small Learning Rate:
    #     Slower learning
    #
    # Large Learning Rate:
    #     Faster learning but may overshoot
    # ============================================================

    learning_rate = 0.1

    # ============================================================
    # ERROR CALCULATION
    # ============================================================
    #
    # Formula:
    #
    # Error =
    # Actual - Predicted
    #
    # Example:
    #
    # Error =
    # 1 - 0.8
    #
    # Error = 0.2
    #
    # Interpretation:
    #
    # Prediction is below the expected output.
    # ============================================================

    error = actual - predicted

    # ============================================================
    # LOSS CALCULATION
    # ============================================================
    #
    # Mean Squared Error (Simplified)
    #
    # Loss = Error²
    #
    # Example:
    #
    # Loss =
    # 0.2²
    #
    # Loss = 0.04
    #
    # Goal:
    #
    # Minimize loss during training.
    # ============================================================

    loss = error ** 2

    # ============================================================
    # GRADIENT CALCULATION
    # ============================================================
    #
    # Gradient tells us:
    #
    # "Which direction should the weight move?"
    #
    # Derivative of MSE:
    #
    # Gradient =
    # 2 × (Predicted - Actual)
    #
    # Example:
    #
    # Gradient =
    # 2 × (0.8 - 1)
    #
    # Gradient = -0.4
    #
    # Negative Gradient:
    # Weight should increase.
    # ============================================================

    gradient = 2 * (predicted - actual)

    # ============================================================
    # WEIGHT UPDATE
    # ============================================================
    #
    # Gradient Descent Rule
    #
    # New Weight =
    # Old Weight -
    # Learning Rate × Gradient
    #
    # Example:
    #
    # New Weight =
    # 0.5 - (0.1 × -0.4)
    #
    # New Weight =
    # 0.54
    #
    # This moves the model toward
    # a better prediction.
    # ============================================================

    updated_weight = weight - learning_rate * gradient

    # ============================================================
    # BUILD RESULTS TABLE
    # ============================================================
    #
    # Display every calculation step used
    # during backpropagation.
    # ============================================================

    df = pd.DataFrame({
        "Metric": [
            "Actual",
            "Predicted",
            "Error",
            "Loss",
            "Gradient",
            "Updated Weight"
        ],
        "Value": [
            actual,
            predicted,
            error,
            loss,
            gradient,
            updated_weight
        ]
    })

    calculation_df = pd.DataFrame({
        "Step": [
            "Error",
            "Loss",
            "Gradient",
            "Weight Update"
        ],
        "Formula": [
            "1 - 0.8",
            "(0.2)^2",
            "2 × (0.8 - 1)",
            "0.5 - (0.1 × -0.4)"
        ],
        "Result": [
            error,
            loss,
            gradient,
            updated_weight
        ]
    })
    # ============================================================
    # BUILD HTML REPORT
    # ============================================================

    html_doc = builder.build_page(
        "Backpropagation Report",
        builder.grid([
            builder.card("Backpropagation Calculations", builder.render_dataframe(df)),
            builder.card("Calculation Steps", builder.render_dataframe(calculation_df)),

            builder.card(
                "Learning Notes",
                builder.render_pre("""
BACKPROPAGATION FLOW

Prediction
     ↓
Calculate Error
     ↓
Calculate Loss
     ↓
Calculate Gradient
     ↓
Update Weight
     ↓
Reduce Future Error


FORMULAS

Error
    = Actual - Predicted

Loss
    = Error²

Gradient
    = 2 × (Predicted - Actual)

Weight Update
    = Weight - Learning Rate × Gradient
                """)
            )
        ])
    )

    # ============================================================
    # SAVE REPORT
    # ============================================================

    ru.save_html_report(
        __file__,
        "backpropagation_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
