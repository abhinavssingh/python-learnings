import pandas as pd

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.deeplearning import (
    ClassificationExperimentRunner,
)
from lib.utility.deeplearning.config.deep_learning_config import DeepLearningConfig
from lib.utility.deeplearning.frameworks.tensorflow.data.tf_text_classification_data_loader import (
    TFTextClassificationDataBundle,
    TFTextClassificationDataLoader,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.sequence.cnn_lstm_wrapper import (
    CNNLSTMWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)
from lib.utility.reports.report_utils import ReportUtils as ru

DATASET_FILE = "GrammarandProductReviews.xlsx"
TEXT_COLUMN = "reviews.text"
RATING_COLUMN = "reviews.rating"
TARGET_COLUMN = "target"

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 150


def _prepare_dataframe() -> pd.DataFrame:
    df = dl.read_dataset(DATASET_FILE, optimize=False, handle_unnamed="drop", return_report=False,)

    required_columns = [TEXT_COLUMN, RATING_COLUMN]
    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    df[RATING_COLUMN] = pd.to_numeric(
        df[RATING_COLUMN],
        errors="coerce",
    )

    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).reset_index(drop=True)

    df[TARGET_COLUMN] = (df[RATING_COLUMN] < 4).astype(int)

    return df


def _run_experiment(
    experiment_name: str,
    df: pd.DataFrame,
    config: DeepLearningConfig,
) -> dict:
    text_data: TFTextClassificationDataBundle = (
        TFTextClassificationDataLoader.from_text_and_target(
            texts=df[TEXT_COLUMN],
            targets=df[TARGET_COLUMN],
            max_nb_words=MAX_NB_WORDS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            test_size=0.2,
            random_state=42,
        )
    )

    model = CNNLSTMWrapper(
        max_nb_words=MAX_NB_WORDS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=50,
        conv_filters=64,
        conv_kernel_size=5,
        pool_size=5,
        lstm_units=64,
        dropout_rate=0.2,
        output_units=2,
        output_activation="softmax",
    )

    utility = TensorFlowModelUtility(model_wrapper=model, config=config,)

    result = ClassificationExperimentRunner.run(
        utility=utility,
        X_train=text_data.X_train,
        y_train=text_data.y_train_onehot,
        validation_data=(text_data.X_test, text_data.y_test_onehot),
        X_test=text_data.X_test,
        y_test=text_data.y_test_onehot,
        y_test_labels=text_data.y_test,
        class_labels=["Good", "Bad"],
        positive_label_index=1,
        experiment_name=experiment_name,
        compile_metrics=["accuracy"],
    )

    return {
        "name": experiment_name,
        "samples": int(len(df)),
        "vocab_size": int(text_data.vocab_size),
        **result,
    }


def main():
    builder = HtmlBuilder()
    plot_renderer = PlotRenderer()
    content = []

    df = _prepare_dataframe()

    info_text = dfh.get_dataframe_info_str(df)

    # Task A: run on subset for quick baseline comparison.
    subset_parts = []

    for class_value in sorted(df[TARGET_COLUMN].unique()):
        class_rows = df[df[TARGET_COLUMN] == class_value]
        subset_parts.append(
            class_rows.sample(frac=0.4, random_state=42)
        )

    subset_df = pd.concat(
        subset_parts,
        axis=0,
    ).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Task B: run on complete dataset.
    full_df = df

    config = DeepLearningConfig(
        epochs=5,
        batch_size=64,
        verbose=0,
        optimizer="adam",
        learning_rate=0.001,
        loss="categorical_crossentropy",
        early_stopping=True,
        patience=2,
        early_stopping_monitor="val_loss",
        reduce_lr=True,
        reduce_lr_factor=0.5,
        reduce_lr_patience=1,
        reduce_lr_monitor="val_loss",
    )

    subset_result = _run_experiment(
        experiment_name="Task A - Subset Dataset",
        df=subset_df,
        config=config,
    )

    full_result = _run_experiment(
        experiment_name="Task B - Full Dataset",
        df=full_df,
        config=config,
    )

    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .rename(index={0: "Good", 1: "Bad"})
        .to_dict()
    )

    quality_summary = {
        "dataset": DATASET_FILE,
        "rows_after_cleaning": int(len(df)),
        "columns": int(df.shape[1]),
        "target_good_count": int(target_distribution.get("Good", 0)),
        "target_bad_count": int(target_distribution.get("Bad", 0)),
        "target_bad_pct": round((target_distribution.get("Bad", 0) / len(df)) * 100, 2),
        "feature_text": TEXT_COLUMN,
        "feature_rating": RATING_COLUMN,
        "max_nb_words": MAX_NB_WORDS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }

    comparison_df = pd.DataFrame(
        [
            {
                "experiment": subset_result["name"],
                "samples": subset_result["samples"],
                "test_loss": subset_result["test_metrics"].get("loss"),
                "test_accuracy": subset_result["test_metrics"].get("accuracy"),
                "train_accuracy": subset_result["final_train_accuracy"],
                "val_accuracy": subset_result["final_val_accuracy"],
                "auc_score": subset_result["auc_score"],
            },
            {
                "experiment": full_result["name"],
                "samples": full_result["samples"],
                "test_loss": full_result["test_metrics"].get("loss"),
                "test_accuracy": full_result["test_metrics"].get("accuracy"),
                "train_accuracy": full_result["final_train_accuracy"],
                "val_accuracy": full_result["final_val_accuracy"],
                "auc_score": full_result["auc_score"],
            },
        ]
    )

    preview_df = df[[TEXT_COLUMN, RATING_COLUMN, TARGET_COLUMN]].head(10)

    content.append(
        builder.grid([
            builder.card("Dataset Info", builder.render_pre(info_text)),
            builder.card("Data Quality Summary", builder.render_dict(quality_summary)),
            builder.card("Task A Metrics", builder.render_dict({
                **subset_result["test_metrics"],
                **subset_result["classification_metrics"],
                "auc_score": subset_result["auc_score"],
                "train_accuracy": subset_result["final_train_accuracy"],
                "val_accuracy": subset_result["final_val_accuracy"],
            })),
            builder.card("Task B Metrics", builder.render_dict({
                **full_result["test_metrics"],
                **full_result["classification_metrics"],
                "auc_score": full_result["auc_score"],
                "train_accuracy": full_result["final_train_accuracy"],
                "val_accuracy": full_result["final_val_accuracy"],
            })),
            builder.card("Sample Prepared Records", builder.render_dataframe(preview_df)),
            builder.card("Task Comparison (Subset vs Full)", builder.render_dataframe(comparison_df)),
            builder.card("Task A - Training History", builder.render_dataframe(subset_result["history_df"])),
            builder.card("Task B - Training History", builder.render_dataframe(full_result["history_df"]))
        ]
        )
    )

    content.append(
        builder.chart_grid([
            plot_renderer.plot_to_card(subset_result["roc_fig"], "Task A - ROC Curve",),
            plot_renderer.plot_to_card(full_result["roc_fig"], "Task B - ROC Curve"),
            plot_renderer.plot_to_card(subset_result["confusion_fig"], "Task A - Confusion Matrix"),
            plot_renderer.plot_to_card(full_result["confusion_fig"], "Task B - Confusion Matrix")
        ])
    )

    content.append(
        builder.full_width_card(
            "CNN-LSTM Model Summary",
            builder.render_pre(full_result["model_summary"], max_visible_lines=70),
        )
    )

    html_doc = builder.build_page(
        "Customer Product Review Classification Using CNN-LSTM",
        "\n".join(content),
    )

    ru.save_html_report(
        __file__,
        "product_review_cnn_lstm_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
