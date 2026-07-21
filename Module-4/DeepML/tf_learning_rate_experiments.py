from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.utility.deeplearning.config.deep_learning_config import (
    DeepLearningConfig,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.mlp_wrapper import (
    MLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_learning_rate_pipeline import (
    TensorFlowLearningRatePipeline,
)
from lib.utility.reports.report_utils import (
    ReportUtils as ru,
)


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
        loss="binary_crossentropy",
    )

    pipeline = TensorFlowLearningRatePipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        learning_rates=[
            1e-1,
            1e-2,
            1e-3,
            1e-4,
            1e-5,
        ],
        model_kwargs={
            "input_dim": 20,
            "output_dim": 1,
            "hidden_layers": [64, 32],
        },
    )

    results_df = pipeline.run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    html_doc = builder.build_page(
        "TensorFlow Learning Rate Comparison Report",
        builder.grid([
            builder.card(
                "Learning Rate Results",
                builder.render_dataframe(results_df),
            ),
            builder.card(
                "Best Learning Rate",
                builder.render_dict(
                    results_df.iloc[0].to_dict()
                ),
            ),
        ])
    )

    ru.save_html_report(
        __file__,
        "tensorflow_learning_rate_experiment_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
