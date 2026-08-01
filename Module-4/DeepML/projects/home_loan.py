import io

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import recall_score

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
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
from lib.utility.deeplearning.visualization.class_distribution_plot import ClassDistributionPlot
from lib.utility.deeplearning.visualization.roc_curve_plot import ROCurvePlot
from lib.utility.deeplearning.visualization.training_history_plot import TrainingHistoryPlot
from lib.utility.reports.report_utils import ReportUtils as ru


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Add compact ratio features to improve signal for repayment behavior.
    df = dfh.insert_column_after(
        df,
        after_col="AMT_INCOME_TOTAL",
        new_col="INCOME_CREDIT_RATIO",
        values=lambda x: x["AMT_INCOME_TOTAL"] / (x["AMT_CREDIT"] + 1.0),
    )

    df = dfh.insert_column_after(
        df,
        after_col="AMT_ANNUITY",
        new_col="ANNUITY_INCOME_RATIO",
        values=lambda x: x["AMT_ANNUITY"] / (x["AMT_INCOME_TOTAL"] + 1.0),
    )

    df = dfh.insert_column_after(
        df,
        after_col="AMT_GOODS_PRICE",
        new_col="CREDIT_GOODS_GAP",
        values=lambda x: x["AMT_CREDIT"] - x["AMT_GOODS_PRICE"],
    )

    df = dfh.insert_column_after(
        df,
        after_col="CNT_FAM_MEMBERS",
        new_col="INCOME_PER_FAMILY_MEMBER",
        values=lambda x: x["AMT_INCOME_TOTAL"] / (x["CNT_FAM_MEMBERS"] + 1.0),
    )

    return df


def main():
    builder = HtmlBuilder()
    plot_renderer = PlotRenderer()
    content = []

    # Optional downsampling keeps local experimentation practical.
    sample_fraction = 0.35

    df, load_report = dl.read_dataset("loan_data.csv", optimize=True, handle_unnamed="drop", return_report=True)

    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    info_text = dfh.get_dataframe_info_str(df)

    missing_df = dfh.check_nan_inf(df)
    missing_df["missing_percent"] = (missing_df["NaN_count"] / len(df) * 100).round(2)
    missing_df = missing_df[missing_df["NaN_count"] > 0].sort_values(
        by="missing_percent",
        ascending=False,
    )

    original_distribution = df["TARGET"].value_counts().sort_index().to_dict()

    default_pct = round((original_distribution.get(1, 0) / len(df)) * 100, 2)
    non_default_pct = round((original_distribution.get(0, 0) / len(df)) * 100, 2)

    imbalance_analysis = {
        "rows_used": len(df),
        "sample_fraction": sample_fraction,
        "target_0_non_default_pct": non_default_pct,
        "target_1_default_pct": default_pct,
        "is_imbalanced": default_pct < 40,
    }

    df = _apply_feature_engineering(df)

    categorical_columns = [
        col
        for col in df.columns
        if col != "TARGET" and not is_numeric_dtype(df[col])
    ]

    if categorical_columns:
        df[categorical_columns] = (
            df[categorical_columns]
            .astype("string")
            .fillna("Unknown")
        )

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
        target_column="TARGET",
        drop_columns=["SK_ID_CURR"],
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

    balanced_distribution = pd.Series(y_train).value_counts().sort_index().to_dict()
    imbalance_summary = preprocess_result.imbalance_summary

    config = DeepLearningConfig(
        epochs=20,
        batch_size=256,
        verbose=0,
        optimizer="adam",
        learning_rate=0.001,
        loss="binary_crossentropy",
        early_stopping=True,
        patience=5,
        early_stopping_monitor="val_loss",
        reduce_lr=True,
        reduce_lr_factor=0.5,
        reduce_lr_patience=2,
        reduce_lr_monitor="val_loss",
    )

    model = MLPWrapper(
        input_dim=X_train.shape[1],
        output_dim=1,
        hidden_layers=[256, 128, 64],
        activation="relu",
        output_activation="sigmoid",
        dropout_rate=0.3,
        initializer="he_normal",
        regularizer="l2",
    )

    utility = TensorFlowModelUtility(model_wrapper=model, config=config,)

    utility.compile(metrics=["accuracy"])

    history = utility.train(X_train=X_train, y_train=y_train, validation_data=(X_test, y_test),)

    tf_metrics = utility.evaluate(X_test=X_test, y_test=y_test,)

    y_prob = utility.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    evaluator = ClassificationEvaluator()
    classification_metrics = evaluator.evaluate(y_true=y_test, y_pred=y_pred,)

    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    roc_fig, auc_score = ROCurvePlot.create_plot(y_true=y_test, y_prob=y_prob)
    history_loss_fig = TrainingHistoryPlot.create_plot(history)
    history_acc_fig = TrainingHistoryPlot.create_accuracy_plot(history)

    all_metrics = {
        **{f"tf_{k}": float(v) for k, v in tf_metrics.items()},
        **{f"cls_{k}": float(v) for k, v in classification_metrics.items()},
        "sensitivity": float(sensitivity),
        "auc_score": float(auc_score),
    }

    model_summary_stream = io.StringIO()

    def _summary_writer(line: str) -> None:
        model_summary_stream.write(line + "\n")

    utility.get_model().summary(print_fn=_summary_writer)

    # content.append(builder.full_width_card("Loan Data",
    #                                       builder.render_dataframe_collapsible(df, initial_rows=15)))

    content.append(
        builder.grid(
            [
                builder.card("Dataset Load Report", builder.render_pre(load_report)),
                builder.card("Dataset Info", builder.render_pre(info_text)),
                builder.card("Target Distribution Analysis", builder.render_dict(imbalance_analysis)),
                builder.card("Feature Engineering Summary", builder.render_dict({
                    "binary_categorical_count": len(binary_categoricals),
                    "one_hot_categorical_count": len(one_hot_categoricals),
                    "final_train_shape": str(X_train.shape),
                    "test_shape": str(X_test.shape),
                })),
                builder.card("Model Evaluation", builder.render_dict(all_metrics)),
                builder.card(
                    "SMOTE Summary",
                    builder.render_dict(imbalance_summary)
                    if imbalance_summary
                    else "No imbalance summary available",
                ),
                builder.card("Missing Value Analysis", builder.render_dataframe(missing_df),
                             ),
                builder.card("Training History", builder.render_dataframe(history.to_dataframe()),
                             )
            ]
        )
    )

    content.append(
        builder.chart_grid([
            plot_renderer.plot_to_card(
                ClassDistributionPlot.create_plot(original_distribution, "Original Train Class Distribution"),
                "Original Train Class Distribution",
            ),
            plot_renderer.plot_to_card(
                ClassDistributionPlot.create_plot(balanced_distribution, "Balanced Train Class Distribution"),
                "Balanced Train Class Distribution",
            ),
            plot_renderer.plot_to_card(history_loss_fig, "Training vs Validation Loss"),
            plot_renderer.plot_to_card(history_acc_fig, "Training vs Validation Accuracy"),
            plot_renderer.plot_to_card(roc_fig, "ROC Curve")
        ])
    )

    content.append(
        builder.full_width_card(
            "Model Summary",
            builder.render_pre(model_summary_stream.getvalue(), max_visible_lines=70),
        )
    )

    html_doc = builder.build_page(
        "Home Loan Prediction - Deep Learning (TensorFlow)",
        "\n".join(content),
    )

    ru.save_html_report(
        __file__,
        "home_loan_dl_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True,
    )


if __name__ == "__main__":
    main()
