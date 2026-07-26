from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder
from lib.utility.deeplearning.config.deep_learning_config import (
    DeepLearningConfig,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.mixed_mlp_wrapper import (
    MixedMLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.mlp_wrapper import (
    MLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.parallel_mlp_wrapper import (
    ParallelMLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_architecture_pipeline import (
    TensorFlowArchitecturePipeline,
)
from lib.utility.reports.report_utils import (
    ReportUtils as ru,
)


def main():

    builder = HtmlBuilder()

    # ==========================================================
    # DATA
    # ==========================================================

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

    # ==========================================================
    # CONFIG
    # ==========================================================

    config = DeepLearningConfig(
        epochs=20,
        batch_size=32,
        optimizer="adam",
        learning_rate=0.001,
        loss="binary_crossentropy",
    )

    # ==========================================================
    # MODELS
    # ==========================================================

    models = [
        MLPWrapper(
            input_dim=20,
            output_dim=1,
            hidden_layers=[64, 32],
            output_activation="sigmoid",
        ),
        ParallelMLPWrapper(
            input_dim=20,
            output_units=1,
            output_activation="sigmoid",
        ),
        MixedMLPWrapper(
            input_dim=20,
            output_units=1,
            output_activation="sigmoid",
        ),
    ]

    # ==========================================================
    # PIPELINE
    # ==========================================================

    pipeline = TensorFlowArchitecturePipeline(
        model_wrappers=models,
        config=config,
    )

    results_df = pipeline.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print(results_df)

    # ==========================================================
    # REPORT
    # ==========================================================

    best_architecture = (
        results_df.iloc[0].to_dict()
        if not results_df.empty
        else {}
    )

    html_doc = builder.build_page(
        "TensorFlow Architecture Comparison Report",
        builder.grid(
            [
                builder.card(
                    "Architecture Comparison Results",
                    builder.render_dataframe(
                        results_df
                    ),
                ),
                builder.card(
                    "Best Architecture",
                    builder.render_dict(
                        best_architecture
                    ),
                ),
            ]
        ),
    )

    ru.save_html_report(
        __file__,
        "tensorflow_architecture_experiment_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
