import io

import numpy as np

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.deeplearning import (
    ReconstructionPlot,
    TrainingHistoryPlot,
)
from lib.utility.deeplearning.config.deep_learning_config import DeepLearningConfig
from lib.utility.deeplearning.frameworks.tensorflow.data.tf_autoencoder_data_loader import (
    TFAutoencoderDataLoader,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.cnn.denoise_autoencoder_wrapper import (
    DenoiseAutoencoderWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)
from lib.utility.reports.report_utils import ReportUtils as ru

DATASET_PATH = "datasets/Dental-Panaromic-Autoencoder.npz"


def main():
    builder = HtmlBuilder()
    plot_renderer = PlotRenderer()
    content = []

    noise_factor = 0.2

    data_bundle = TFAutoencoderDataLoader.build_noisy_bundle(
        npz_path=DATASET_PATH,
        noise_factor=noise_factor,
        seed=42,
        to_grayscale=True,
    )

    input_shape = tuple(data_bundle.x_train_clean.shape[1:])

    config = DeepLearningConfig(
        epochs=50,
        batch_size=16,
        verbose=0,
        optimizer="adam",
        learning_rate=0.001,
        loss="mse",
        early_stopping=True,
        patience=8,
        early_stopping_monitor="val_loss",
        reduce_lr=True,
        reduce_lr_factor=0.5,
        reduce_lr_patience=4,
        reduce_lr_monitor="val_loss",
    )

    model_wrapper = DenoiseAutoencoderWrapper(
        input_shape=input_shape,
    )

    utility = TensorFlowModelUtility(
        model_wrapper=model_wrapper,
        config=config,
    )

    utility.compile(metrics=["mae"])

    history = utility.train(
        X_train=data_bundle.x_train_noisy,
        y_train=data_bundle.x_train_clean,
        validation_data=(
            data_bundle.x_test_noisy,
            data_bundle.x_test_clean,
        ),
    )

    test_metrics = utility.evaluate(
        X_test=data_bundle.x_test_noisy,
        y_test=data_bundle.x_test_clean,
    )

    reconstructed = utility.predict(data_bundle.x_test_noisy)

    loss_curve = TrainingHistoryPlot.create_metric_plot(
        history=history,
        train_metric="loss",
        val_metric="val_loss",
        title="Training vs Validation Loss",
        yaxis_title="Loss",
    )

    mae_curve = TrainingHistoryPlot.create_metric_plot(
        history=history,
        train_metric="mae",
        val_metric="val_mae",
        title="Training vs Validation MAE",
        yaxis_title="MAE",
    )

    original_grid = ReconstructionPlot.create_grid(
        images=data_bundle.x_train_clean,
        title="First 5 Original X-ray Images",
        max_images=5,
        columns=5,
    )

    noisy_grid = ReconstructionPlot.create_grid(
        images=data_bundle.x_train_noisy,
        title="First 5 Noisy X-ray Images",
        max_images=5,
        columns=5,
    )

    reconstruction_grid = ReconstructionPlot.create_noisy_vs_reconstructed(
        noisy_images=data_bundle.x_test_noisy,
        reconstructed_images=reconstructed,
        max_images=10,
    )

    model_summary_stream = io.StringIO()

    def _summary_writer(line: str) -> None:
        model_summary_stream.write(line + "\n")

    utility.get_model().summary(print_fn=_summary_writer)

    dataset_summary = {
        "dataset_path": DATASET_PATH,
        "x_train_clean_shape": str(data_bundle.x_train_clean.shape),
        "x_test_clean_shape": str(data_bundle.x_test_clean.shape),
        "input_shape": str(input_shape),
        "noise_factor": noise_factor,
    }

    evaluation_summary = {
        "test_loss": float(test_metrics.get("loss", np.nan)),
        "test_mae": float(test_metrics.get("mae", np.nan)),
    }

    content.append(
        builder.grid([
            builder.card("Dataset Summary", builder.render_dict(dataset_summary)),
            builder.card("Evaluation Summary", builder.render_dict(evaluation_summary)),
            builder.card("Training Configuration",
                         builder.render_dict(
                             {
                                 "epochs": config.epochs,
                                 "batch_size": config.batch_size,
                                 "optimizer": config.optimizer,
                                 "loss": config.loss,
                                 "metrics": "mae",
                             }
                         )),
            builder.card("Training History Table", builder.render_dataframe(history.to_dataframe()),
                         )
        ])
    )

    content.append(
        builder.full_width_card(
            "Model Summary",
            builder.render_pre(model_summary_stream.getvalue(), max_visible_lines=80),
        )
    )

    content.append(builder.chart_grid([
        plot_renderer.plot_to_card(loss_curve, "Loss Curve"),
        plot_renderer.plot_to_card(mae_curve, "MAE Curve"),
    ]))

    content.append(
        plot_renderer.plot_to_full_width_card(original_grid, "Original Images")
    )

    content.append(
        plot_renderer.plot_to_full_width_card(noisy_grid, "Noisy Images")
    )

    content.append(
        plot_renderer.plot_to_full_width_card(reconstruction_grid, "Reconstructed Images")
    )

    html_doc = builder.build_page(
        "Enhancing Dental X-rays Using Autoencoders",
        "\n".join(content),
    )

    ru.save_html_report(
        __file__,
        "dental_autoencoder_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,


    )


if __name__ == "__main__":
    main()
