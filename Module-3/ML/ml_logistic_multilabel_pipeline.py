import time

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.machinelearning.facade.ClassificationModelUtility import ClassificationModelUtility as cmu
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.generic.ClassificationPlots import ClassificationPlots as cp
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running Multi-label Classification Pipeline...")

    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plotRenderer = PlotRenderer()
    cplots = cp()

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df, report = dl.read_dataset("adultcensusincome.csv", optimize=False, handle_unnamed="drop", return_report=True)

    # ✅ ALWAYS reset after filtering (CRITICAL)
    df_usa = df[df['native.country'] == 'United-States'].copy().reset_index(drop=True)

    # ✅ CREATE LABELS
    df_usa["labels"] = df_usa.apply(
        lambda x: list(filter(None, [

            # income
            "high_income" if x["income"] == ">50K" else None,

            # education
            "high_edu" if x["education.num"] >= 10 else None,
            "low_edu" if x["education.num"] < 6 else None,

            # age group
            "senior" if x["age"] > 50 else None,
            "young" if x["age"] < 30 else None,

            # work class
            "private_emp" if x["workclass"] == "Private" else None,

        ])), axis=1
    )

    # ✅ MULTI-LABEL ENCODING
    mlb = MultiLabelBinarizer()

    y = pd.DataFrame(mlb.fit_transform(df_usa["labels"]), columns=mlb.classes_).reset_index(drop=True)

    # ✅ REMOVE BAD LABELS (CRITICAL)
    y = y.loc[:, y.nunique() > 1]

    # ✅ FEATURES
    X = df_usa.drop(columns=["labels", "income"]).reset_index(drop=True)

    # ✅ CONCAT SAFE
    df_ml = pd.concat([X, y], axis=1)

    # ✅ EXTRA SAFETY (no harm)
    df_ml[y.columns] = df_ml[y.columns].fillna(0).astype(int)

    target_cols = list(y.columns)

    # ---------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------
    imputer = CustomImputer(
        num_strategy="mode",
        groupby_cols=["native.country", "workclass", "occupation"]
    )

    outlier = OutlierHandler(method="iqr", factor=1.5)

    # ---------------------------------------------------
    # INIT CLASSIFICATION UTILITY
    # ---------------------------------------------------
    cm = cmu(
        df_ml,
        target_col=target_cols,   # ✅ MULTI-LABEL SUPPORT
        imputer=imputer,
        outlier_handler=outlier
    )

    cm.prepare_data()

    # ---------------------------------------------------
    # RUN MODELS
    # ---------------------------------------------------
    results = cm.run_all_models()
    results_df = cm.get_results_df()
    print(results_df.columns.tolist())

    # ---------------------------------------------------
    # MODEL ANALYSIS
    # ---------------------------------------------------
    ranked = cm.rank_models(metric="f1")
    best_model = cm.get_best_model(metric="f1")

    # ---------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------
    content.append(builder.grid([
        builder.card("Results", builder.render_dataframe(results_df)),
        builder.card("Best Model", builder.render_dict(best_model)),
    ]))

    content.append(builder.chart_grid([
        plotRenderer.plot_to_card(cplots.plot_bar(results_df, metric="f1"), "F1 Score Comparison"),
        plotRenderer.plot_to_card(cplots.plot_multi_metrics(results_df), "Multi-Metric Comparison"),
        # plotRenderer.plot_to_card(cplots.plot_roc_all_models(cm.results), "ROC Curve Comparison"),
    ]))

    # ---------------------------------------------------
    # REPORT
    # ---------------------------------------------------
    html_doc = builder.build_page(
        "Multi-label Classification Report",
        "\n".join(content)
    )

    output_path = ru.save_html_report(
        __file__,
        "ml_multilabel_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )

    print(f"Wrote report to: {output_path}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
