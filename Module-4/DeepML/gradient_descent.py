import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    """
    Demonstration of Gradient Descent Optimization.

    Goal:
        Find the minimum value of the loss function:

            Loss = w²

    The minimum occurs at:

            w = 0

    Gradient Descent works by repeatedly moving the weight
    in the direction opposite to the gradient.

            New Weight =
                Current Weight
                - Learning Rate × Gradient
    """

    builder = HtmlBuilder()

    # Initial weight value.
    # Think of this as a model parameter that needs optimization.
    weight = 5.0

    # Controls how large each update step will be.
    #
    # Small value  -> Slow convergence
    # Large value  -> Faster learning but may overshoot
    learning_rate = 0.1

    # Store training history so we can later visualize:
    # Epoch, Weight, Loss, Gradient
    history = []

    # Run optimization for 20 iterations (epochs)
    for epoch in range(20):

        # ---------------------------------------------------------
        # Loss Function
        #
        #     Loss = w²
        #
        # Our objective is to minimize this value.
        #
        # Examples:
        #
        #     w = 5  -> Loss = 25
        #     w = 2  -> Loss = 4
        #     w = 0  -> Loss = 0 (minimum)
        # ---------------------------------------------------------
        loss = weight ** 2

        # ---------------------------------------------------------
        # Gradient Calculation
        #
        # Loss = w²
        #
        # Derivative:
        #
        # d(Loss)/dw = 2w
        #
        # The gradient tells us:
        #
        # - Which direction to move
        # - How steep the slope is
        #
        # Large gradient  -> Large updates
        # Small gradient  -> Small updates
        # ---------------------------------------------------------
        gradient = 2 * weight

        # Save current state for reporting and visualization.
        history.append([
            epoch + 1,
            weight,
            loss,
            gradient
        ])

        # ---------------------------------------------------------
        # Gradient Descent Update Rule
        #
        # New Weight =
        #       Current Weight
        #       - Learning Rate × Gradient
        #
        # Example:
        #
        #     weight = 5.0
        #     gradient = 10.0
        #     learning_rate = 0.1
        #
        #     new_weight =
        #         5.0 - (0.1 × 10)
        #
        #         5.0 - 1.0
        #
        #         4.0
        #
        # As training continues, the weight moves closer to 0,
        # reducing the loss.
        # Gradient Descent Journey

        #    Weight = 5.0

        #    Epoch 1:
        #        Loss = 25.0
        #        ↓

        #    Epoch 2:
        #        Weight = 4.0
        #        Loss = 16.0
        #        ↓

        #    Epoch 3:
        #        Weight = 3.2
        #        Loss = 10.24
        #        ↓

        #    Epoch 4:
        #        Weight = 2.56
        #        Loss = 6.55
        #        ↓

        #    Eventually:

        #        Weight → 0
        #        Loss   → 0
        # ---------------------------------------------------------

        weight = weight - learning_rate * gradient

    # Convert optimization history to a DataFrame
    # so it can be displayed in the HTML report.
    df = pd.DataFrame(
        history,
        columns=[
            "Epoch",
            "Weight",
            "Loss",
            "Gradient"
        ]
    )

    # Build report
    html_doc = builder.build_page(
        "Gradient Descent Report",
        builder.grid([
            builder.card(
                "Optimization History",
                builder.render_dataframe(df)
            )
        ])
    )

    # Save report to:
    #
    # reports/gradient_descent_report.html
    #
    ru.save_html_report(
        __file__,
        "gradient_descent_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
