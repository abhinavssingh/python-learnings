from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.utility.deeplearning.config.deep_learning_config import (
    DeepLearningConfig,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.mlp_wrapper import (
    MLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_model_pipeline import (
    TensorFlowModelPipeline,
)
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    builder = HtmlBuilder()
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    config = DeepLearningConfig(
        epochs=20,
        batch_size=32,
        optimizer="adam",
        learning_rate=0.001,
        loss="binary_crossentropy",
    )

    model = MLPWrapper(
        input_dim=20,
        output_dim=1,
        hidden_layers=[64, 32],
    )

    pipeline = TensorFlowModelPipeline(
        model_wrapper=model,
        config=config,
    )

    result = pipeline.run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print(result.metrics)

    # ==========================================================
    # REPORT
    # ==========================================================

    html_doc = builder.build_page(
        "TensorFlow Model Experiment Report",
        builder.grid([
            builder.card("Model Name", builder.render_pre(result.model_name)),
            builder.card("Predictions", builder.render_dict(result.artifacts)),
            builder.card("Training Metrics", builder.render_dict(result.metrics)),
            builder.card("Training History", builder.render_dict(result.history.to_dict())),
        ]))

    ru.save_html_report(
        __file__,
        "tensorflow_model_experiment_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
