import io

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.deeplearning import (
    BarChartPlot,
    ClassDistributionPlot,
    ConfusionMatrixPlot,
    HeatmapPlot,
    HistogramPlot,
    ROCurvePlot,
    TrainingHistoryPlot,
)
from lib.utility.deeplearning.config.deep_learning_config import DeepLearningConfig
from lib.utility.deeplearning.evaluation.classification_evaluator import (
    ClassificationEvaluator,
)
from lib.utility.deeplearning.frameworks.tensorflow.models.dense.mlp_wrapper import (
    MLPWrapper,
)
from lib.utility.deeplearning.frameworks.tensorflow.tensorflow_model_utility import (
    TensorFlowModelUtility,
)
from lib.utility.deeplearning.preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from lib.utility.reports.report_utils import ReportUtils as ru

TARGET_COLUMN = "not.fully.paid"
DATASET_FILE = "loan_data (2).csv"


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert log annual income into linear scale to enable richer ratios.
    df = dfh.insert_column_after(
        df,
        after_col="log.annual.inc",
        new_col="annual.inc",
        values=lambda x: np.exp(np.clip(x["log.annual.inc"], -10, 20)),
    )

    df = dfh.insert_column_after(
        df,
        after_col="installment",
        new_col="installment_income_ratio",
        values=lambda x: x["installment"] / (x["annual.inc"] + 1.0),
    )

    df = dfh.insert_column_after(
        df,
        after_col="revol.bal",
        new_col="revolbal_income_ratio",
        values=lambda x: x["revol.bal"] / (x["annual.inc"] + 1.0),
    )

    return df


def _high_correlation_columns(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = 0.9,
) -> list[str]:
    numeric_cols = [
        col
        for col in df.columns
        if col != target_column and is_numeric_dtype(df[col])
    ]

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > threshold)
    ]

    return to_drop


def main():
    builder = HtmlBuilder()
    plot_renderer = PlotRenderer()
    content = []

    df, load_report = dl.read_dataset(DATASET_FILE, optimize=True, handle_unnamed="drop", return_report=True)

    info_text = dfh.get_dataframe_info_str(df)

    duplicate_count = int(df.duplicated().sum())

    missing_df = dfh.check_nan_inf(df)
    missing_df["missing_percent"] = (missing_df["NaN_count"] / len(df) * 100).round(2)
    missing_df = missing_df[missing_df["NaN_count"] > 0].sort_values(
        by="missing_percent",
        ascending=False,
    )

    original_distribution = df[TARGET_COLUMN].value_counts().sort_index()
    default_pct = round((original_distribution.get(1, 0) / len(df)) * 100, 2)
    non_default_pct = round((original_distribution.get(0, 0) / len(df)) * 100, 2)

    quality_summary = {
        "dataset": DATASET_FILE,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_count,
        "non_default_pct": non_default_pct,
        "default_pct": default_pct,
        "is_imbalanced": default_pct < 40,
    }

    df = _feature_engineering(df)

    correlation_columns_dropped = _high_correlation_columns(
        df,
        target_column=TARGET_COLUMN,
        threshold=0.9,
    )

    corr_numeric = df[[
        col for col in df.columns
        if is_numeric_dtype(df[col])
    ]].corr().round(3)

    categorical_columns = [
        col
        for col in df.columns
        if col != TARGET_COLUMN and not is_numeric_dtype(df[col])
    ]

    if categorical_columns:
        df[categorical_columns] = df[categorical_columns].astype("string").fillna("Unknown")

    binary_categoricals = [
        col for col in categorical_columns
        if df[col].nunique(dropna=False) <= 2
    ]

    one_hot_categoricals = [
        col for col in categorical_columns
        if col not in binary_categoricals
    ]

    preprocess_result = DataPreprocessor.prepare_classification_data(
        df=df,
        target_column=TARGET_COLUMN,
        drop_columns=correlation_columns_dropped,
        label_encode_columns=binary_categoricals,
        one_hot_columns=one_hot_categoricals,
        scale_numeric=True,
        test_size=0.2,
        random_state=42,
        oversample=True,
        oversample_method="smote",
        oversample_params={
            "sampling_strategy": "auto",
            "k_neighbors": 5,
            "random_state": 42,
        },
    )

    X_train = preprocess_result.X_train
    y_train = np.asarray(preprocess_result.y_train)
    X_test = preprocess_result.X_test
    y_test = np.asarray(preprocess_result.y_test)

    balanced_distribution = pd.Series(y_train).value_counts().sort_index()

    config = DeepLearningConfig(
        epochs=40,
        batch_size=128,
        verbose=0,
        optimizer="adam",
        learning_rate=0.001,
        loss="binary_crossentropy",
        early_stopping=True,
        patience=6,
        early_stopping_monitor="val_loss",
        reduce_lr=True,
        reduce_lr_factor=0.5,
        reduce_lr_patience=3,
        reduce_lr_monitor="val_loss",
    )

    model = MLPWrapper(
        input_dim=X_train.shape[1],
        output_dim=1,
        hidden_layers=[128, 64, 32],
        activation="relu",
        output_activation="sigmoid",
        dropout_rate=0.25,
        initializer="he_normal",
        regularizer="l2",
    )

    utility = TensorFlowModelUtility(
        model_wrapper=model,
        config=config,
    )

    utility.compile(metrics=["accuracy"])

    history = utility.train(
        X_train=X_train,
        y_train=y_train,
        validation_data=(X_test, y_test),
    )

    tf_metrics = utility.evaluate(X_test=X_test, y_test=y_test)

    y_prob = utility.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    evaluator = ClassificationEvaluator()
    cls_metrics = evaluator.evaluate(y_true=y_test, y_pred=y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    roc_fig, auc_score = ROCurvePlot.create_plot(y_test, y_prob)

    metrics_summary = {
        **{f"tf_{k}": float(v) for k, v in tf_metrics.items()},
        **{f"cls_{k}": float(v) for k, v in cls_metrics.items()},
        "auc_score": float(auc_score),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    feature_engineering_summary = {
        "engineered_features": "annual.inc, installment_income_ratio, revolbal_income_ratio",
        "categorical_columns": ", ".join(categorical_columns) if categorical_columns else "None",
        "dropped_high_correlation_columns": ", ".join(correlation_columns_dropped) if correlation_columns_dropped else "None",
        "train_shape_after_balancing": str(X_train.shape),
        "test_shape": str(X_test.shape),
    }

    model_summary_stream = io.StringIO()

    def _summary_writer(line: str) -> None:
        model_summary_stream.write(line + "\n")

    utility.get_model().summary(print_fn=_summary_writer)

    purpose_default_rate = (
        df.groupby("purpose")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .round(4)
    )

    content.append(
        builder.grid([
            builder.card("Dataframe Optimization Report", builder.render_pre(load_report)),
            builder.card("Data Quality", builder.render_pre(info_text)),
            builder.card("Quality Summary", builder.render_dict(quality_summary)),
            builder.card("Feature Engineering", builder.render_dict(feature_engineering_summary)),
            builder.card("Evaluation Metrics", builder.render_dict(metrics_summary)),
            builder.card("Missing Value Analysis",
                         builder.render_dataframe(missing_df if not missing_df.empty else pd.DataFrame({"message": ["No missing values"]})),
                         )
        ])
    )

    content.append(builder.chart_grid([
        plot_renderer.plot_to_card(
            ClassDistributionPlot.create_plot(original_distribution.to_dict(), "Original Class Distribution"),
            "Original Class Distribution",
        ),
        plot_renderer.plot_to_card(
            ClassDistributionPlot.create_plot(balanced_distribution.to_dict(), "Balanced Train Class Distribution"),
            "Balanced Train Class Distribution",
        ),
        plot_renderer.plot_to_card(
            BarChartPlot.create_plot(df["purpose"].value_counts(), "Loan Purpose Distribution", "Purpose", "Count"),
            "Loan Purpose Distribution",
        ),
        plot_renderer.plot_to_card(
            BarChartPlot.create_plot(purpose_default_rate, "Default Rate by Purpose", "Purpose", "Default Rate"),
            "Default Rate by Purpose",
        ),
        plot_renderer.plot_to_card(
            HistogramPlot.create_plot(df["int.rate"], "Interest Rate Distribution", "int.rate"),
            "Interest Rate Distribution",
        ),
        plot_renderer.plot_to_card(
            HistogramPlot.create_plot(df["fico"], "FICO Score Distribution", "fico"),
            "FICO Score Distribution",
        ),
        plot_renderer.plot_to_card(
            HeatmapPlot.create_plot(corr_numeric, "Correlation Matrix"),
            "Correlation Heatmap (Numeric Features)",
        ),
        plot_renderer.plot_to_card(roc_fig, "ROC Curve"),
        plot_renderer.plot_to_card(
            TrainingHistoryPlot.create_plot(history),
            "Training History",
        ),
        plot_renderer.plot_to_card(
            ConfusionMatrixPlot.create_plot(y_test, y_pred, class_labels=["Non-default", "Default"]),
            "Confusion Matrix",
        )
    ]))

    content.append(
        builder.full_width_card(
            "Model Summary",
            builder.render_pre(model_summary_stream.getvalue(), max_visible_lines=70),
        )
    )

    html_doc = builder.build_page(
        "Lending Club Loan Prediction - TensorFlow Deep Learning",
        "\n".join(content),
    )

    ru.save_html_report(
        __file__,
        "lending_club_dl_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
