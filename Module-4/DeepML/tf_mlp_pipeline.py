"""
Deep Learning Pipeline Example

This example demonstrates:

1. Dataset Creation
2. Model Creation using MLPWrapper
3. Training using DeepLearningModelUtility
4. Evaluation
5. Reporting
5. Visualization

Architecture:

Input Layer
      ↓
Dense(64, ReLU)
      ↓
Dense(32, ReLU)
      ↓
Dense(1, Sigmoid)

Problem:
    Binary Classification
"""

import pandas as pd
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.html.plotrenderer import PlotRenderer
from lib.utility.deeplearning.config.deep_learning_config import DeepLearningConfig
from lib.utility.deeplearning.tensorflow.models.mixed_mlp_wrapper import MixedMLPWrapper
from lib.utility.deeplearning.tensorflow.models.mlp_wrapper import MLPWrapper
from lib.utility.deeplearning.tensorflow.models.parallel_mlp_wrapper import ParallelMLPWrapper
from lib.utility.deeplearning.tensorflow.tensorflow_model_utility import TensorFlowModelUtility
from lib.utility.deeplearning.visualization.training_history_plot import TrainingHistoryPlot
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    builder = HtmlBuilder()
    plotrenderer = PlotRenderer()
    content = []

    # ==========================================
    # Create Sample Dataset
    # ==========================================

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # ==========================================
    # Configuration
    # ==========================================

    config = DeepLearningConfig(
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        validation_split=0.2
    )

    # ==========================================
    # Build Networks
    # ==========================================

    wrappers = [
        MLPWrapper(
            input_dim=X_train.shape[1],
            hidden_layers=[64, 32],
            output_units=1,
            output_activation="sigmoid"
        ),
        ParallelMLPWrapper(
            input_dim=X_train.shape[1]
        ),
        MixedMLPWrapper(
            input_dim=X_train.shape[1],
            output_units=1,
            output_activation="sigmoid"
        )
    ]

    comparison_results = []

    all_reports = []

    # ==========================================
    # Train All Models
    # ==========================================

    for wrapper in wrappers:

        print(f"\nTraining {wrapper.model_name}")

        dl_util = TensorFlowModelUtility(
            model_wrapper=wrapper,
            config=config
        )

        dl_util.compile(
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # ==========================================
        # Train
        # ==========================================

        history = dl_util.train(
            X_train,
            y_train
        )

        # ==========================================
        # Evaluate
        # ==========================================

        evaluation = dl_util.evaluate(
            X_test,
            y_test
        )

        # ==========================================
        # Predict
        # ==========================================

        predictions = dl_util.predict(X_test)

        predicted_classes = (
            predictions > 0.5
        ).astype(int)

        # ==========================================
        # Classification Report
        # ==========================================

        report_dict = classification_report(
            y_test,
            predicted_classes,
            output_dict=True
        )

        # ==========================================
        # Capture Comparison Results
        # ==========================================

        comparison_results.append({
            "Model": wrapper.model_name,
            "Loss": round(float(evaluation[0]), 4),
            "Accuracy": round(float(evaluation[1]), 4),
            "Precision": round(report_dict["weighted avg"]["precision"], 4),
            "Recall": round(report_dict["weighted avg"]["recall"], 4),
            "F1": round(report_dict["weighted avg"]["f1-score"], 4)
        })

        # ==========================================
        # Individual Report Data
        # ==========================================

        report_df = pd.DataFrame(
            report_dict
        ).transpose()

        history_df = pd.DataFrame(
            history.history
        )

        history_df.insert(
            0,
            "Epoch",
            range(1, len(history_df) + 1)
        )

        all_reports.append({
            "model_name": wrapper.model_name,
            "report_df": report_df,
            "history_df": history_df,
            "history": history
        })

    # ==========================================
    # Comparison DataFrame
    # ==========================================

    comparison_df = pd.DataFrame(
        comparison_results
    )

    # ==========================================
    # Comparison Summary
    # ==========================================

    content.append(builder.card("Model Comparison", builder.render_dataframe(comparison_df)))

    # ==========================================
    # Individual Model Reports
    # ==========================================

    report_cards = []

    for model_report in all_reports:

        model_name = model_report["model_name"]

        report_df = model_report["report_df"]

        history_df = model_report["history_df"]

        report_cards.extend([
            builder.card(f"{model_name} - Classification Report", builder.render_dataframe(report_df)),
            builder.card(f"{model_name} - Training History", builder.render_dataframe(history_df))
        ])

    content.append(builder.grid(report_cards))

    plot_cards = []

    for model_report in all_reports:

        model_name = model_report["model_name"]

        history = model_report["history"]

        loss_plot = TrainingHistoryPlot.create_plot(history)

        accuracy_plot = TrainingHistoryPlot.create_accuracy_plot(history)

        plot_cards.extend([
            plotrenderer.plot_to_card(
                loss_plot,
                f"{model_name} - Loss Curve"
            ),
            plotrenderer.plot_to_card(
                accuracy_plot,
                f"{model_name} - Accuracy Curve"
            )
        ])

    content.append(
        builder.chart_grid(plot_cards)
    )

    compare_fig = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Model Accuracy Comparison"
    )

    content.append(plotrenderer.plot_to_card(compare_fig, "Model Comparison"))

    html_doc = builder.build_page(
        "TensorFlow MLP Training Pipeline Report",
        "\n".join(content)
    )

    # ==========================================
    # Save Report
    # ==========================================

    ru.save_html_report(
        __file__,
        "tf_mlp_training_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
