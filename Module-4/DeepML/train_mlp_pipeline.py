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
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.utility.deeplearning.config.deep_learning_config import DeepLearningConfig
from lib.utility.deeplearning.deep_learning_model_utility import DeepLearningModelUtility
from lib.utility.deeplearning.models.mlp_wrapper import MLPWrapper
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    builder = HtmlBuilder()

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
    # Build Network
    # ==========================================

    wrapper = MLPWrapper(
        input_dim=X_train.shape[1],
        hidden_layers=[64, 32],
        output_units=1,
        output_activation="sigmoid"
    )

    # ==========================================
    # Utility
    # ==========================================

    dl_util = DeepLearningModelUtility(model_wrapper=wrapper, config=config)

    # ==========================================
    # Compile
    # ==========================================

    dl_util.compile(loss="binary_crossentropy", metrics=["accuracy"])

    # ==========================================
    # Train
    # ==========================================

    history = dl_util.train(X_train, y_train)

    # ==========================================
    # Evaluate
    # ==========================================

    evaluation = dl_util.evaluate(X_test, y_test)

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

    report_dict = classification_report(y_test, predicted_classes, output_dict=True)

    report_df = pd.DataFrame(report_dict).transpose()

    # ==========================================
    # History DataFrame
    # ==========================================

    history_df = pd.DataFrame(history.history)

    history_df.insert(0, "Epoch", range(1, len(history_df) + 1))

    # ==========================================
    # Evaluation Summary
    # ==========================================

    evaluation_df = pd.DataFrame(
        [{
            "Loss": evaluation[0],
            "Accuracy": evaluation[1]
        }]
    )

    # ==========================================
    # HTML Report
    # ==========================================

    html_doc = builder.build_page(
        "Deep Learning MLP Training Report",

        builder.grid([
            builder.card("Evaluation Summary", builder.render_dataframe(evaluation_df)),
            builder.card("Classification Report", builder.render_dataframe(report_df)),
            builder.card("Training History", builder.render_dataframe(history_df))
        ])
    )

    # ==========================================
    # Save Report
    # ==========================================

    ru.save_html_report(
        __file__,
        "mlp_training_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
