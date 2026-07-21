from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.utility.deeplearning.config.deep_learning_config import (
    DeepLearningConfig,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.mlp_wrapper import (
    MLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_dropout_pipeline import (
    TensorFlowDropoutPipeline,
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
        learning_rate=0.001,
        loss="binary_crossentropy",
    )

    pipeline = TensorFlowDropoutPipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        dropout_rates=[
            0.0,
            0.1,
            0.2,
            0.3,
            0.5,
        ],
        model_kwargs={
            "input_dim": 20,
            "output_dim": 1,
            "output_activation": "sigmoid",
        },
    )

    results_df = pipeline.run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    html_doc = builder.build_page(
        "TensorFlow Dropout Comparison Report",
        builder.grid([
            builder.card(
                "Dropout Results",
                builder.render_dataframe(results_df),
            ),
            builder.card(
                "Best Dropout Rate",
                builder.render_dict(
                    results_df.iloc[0].to_dict()
                ),
            ),
        ])
    )

    ru.save_html_report(
        __file__,
        "tensorflow_dropout_experiment_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
