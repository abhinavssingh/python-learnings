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
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_activation_pipeline import (
    TensorFlowActivationPipeline,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_architecture_pipeline import (
    TensorFlowArchitecturePipeline,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_batch_size_pipeline import (
    TensorFlowBatchSizePipeline,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_dropout_pipeline import (
    TensorFlowDropoutPipeline,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_initializer_pipeline import (
    TensorFlowInitializerPipeline,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_learning_rate_pipeline import (
    TensorFlowLearningRatePipeline,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_optimizer_pipeline import (
    TensorFlowOptimizerPipeline,
)
from lib.utility.deeplearning.frameworks.tensorflow.pipelines.tf_regularization_pipeline import (
    TensorFlowRegularizationPipeline,
)
from lib.utility.reports.report_utils import ReportUtils as ru


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
    # BASE CONFIG
    # ==========================================================

    config = DeepLearningConfig(
        epochs=20,
        batch_size=32,
        optimizer="adam",
        learning_rate=0.001,
        loss="binary_crossentropy",
    )

    # ==========================================================
    # MODEL KWARGS
    # ==========================================================

    model_kwargs = {
        "input_dim": 20,
        "output_dim": 1,
        "hidden_layers": [64, 32],
        "output_activation": "sigmoid",
    }

    # ==========================================================
    # ARCHITECTURE
    # ==========================================================

    architecture_df = TensorFlowArchitecturePipeline(
        model_wrappers=[
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
        ],
        config=config,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # ACTIVATION
    # ==========================================================

    activation_df = TensorFlowActivationPipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        activations=[
            "relu",
            "tanh",
            "sigmoid",
            "elu",
            "selu",
            "gelu",
        ],
        model_kwargs=model_kwargs,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # OPTIMIZER
    # ==========================================================

    optimizer_df = TensorFlowOptimizerPipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        optimizers=[
            "adam",
            "adamw",
            "sgd",
            "rmsprop",
            "adagrad",
        ],
        model_kwargs=model_kwargs,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # LEARNING RATE
    # ==========================================================

    learning_rate_df = TensorFlowLearningRatePipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        learning_rates=[
            1e-1,
            1e-2,
            1e-3,
            1e-4,
            1e-5,
        ],
        model_kwargs=model_kwargs,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # INITIALIZER
    # ==========================================================

    initializer_df = TensorFlowInitializerPipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        initializers=[
            "he_normal",
            "he_uniform",
            "glorot_normal",
            "glorot_uniform",
        ],
        model_kwargs=model_kwargs,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # BATCH SIZE
    # ==========================================================

    batch_size_df = TensorFlowBatchSizePipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        batch_sizes=[
            16,
            32,
            64,
            128,
            256,
        ],
        model_kwargs=model_kwargs,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # REGULARIZATION
    # ==========================================================

    regularization_df = TensorFlowRegularizationPipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        regularizations=[
            None,
            "l1",
            "l2",
            "l1_l2",
        ],
        model_kwargs=model_kwargs,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # DROPOUT
    # ==========================================================

    dropout_df = TensorFlowDropoutPipeline(
        model_wrapper_class=MLPWrapper,
        base_config=config,
        dropout_rates=[
            0.0,
            0.1,
            0.2,
            0.3,
            0.5,
        ],
        model_kwargs=model_kwargs,
    ).run(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ==========================================================
    # BEST CONFIGURATION
    # ==========================================================

    best_summary = {
        "Architecture":
            architecture_df.iloc[0].to_dict()
            if not architecture_df.empty else {},
        "Activation":
            activation_df.iloc[0].to_dict()
            if not activation_df.empty else {},
        "Optimizer":
            optimizer_df.iloc[0].to_dict()
            if not optimizer_df.empty else {},
        "Learning Rate":
            learning_rate_df.iloc[0].to_dict()
            if not learning_rate_df.empty else {},
        "Initializer":
            initializer_df.iloc[0].to_dict()
            if not initializer_df.empty else {},
        "Batch Size":
            batch_size_df.iloc[0].to_dict()
            if not batch_size_df.empty else {},
        "Regularization":
            regularization_df.iloc[0].to_dict()
            if not regularization_df.empty else {},
        "Dropout":
            dropout_df.iloc[0].to_dict()
            if not dropout_df.empty else {},
    }

    # ==========================================================
    # HTML REPORT
    # ==========================================================

    html_doc = builder.build_page(
        "TensorFlow Comprehensive Benchmark Report",
        builder.grid(
            [
                builder.card(
                    "Best Overall Configuration",
                    builder.render_dict(best_summary),
                ),
                builder.card(
                    "Architecture Results",
                    builder.render_dataframe(architecture_df),
                ),
                builder.card(
                    "Activation Results",
                    builder.render_dataframe(activation_df),
                ),
                builder.card(
                    "Optimizer Results",
                    builder.render_dataframe(optimizer_df),
                ),
                builder.card(
                    "Learning Rate Results",
                    builder.render_dataframe(learning_rate_df),
                ),
                builder.card(
                    "Initializer Results",
                    builder.render_dataframe(initializer_df),
                ),
                builder.card(
                    "Batch Size Results",
                    builder.render_dataframe(batch_size_df),
                ),
                builder.card(
                    "Regularization Results",
                    builder.render_dataframe(regularization_df),
                ),
                builder.card(
                    "Dropout Results",
                    builder.render_dataframe(dropout_df),
                ),
            ]
        ),
    )

    ru.save_html_report(
        __file__,
        "tensorflow_comprehensive_experiment_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
