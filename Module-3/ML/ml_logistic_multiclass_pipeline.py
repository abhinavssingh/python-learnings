import time

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.ClassificationModelUtility import ClassificationModelUtility as cmu
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.generic.ClassificationPlots import ClassificationPlots as cp
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running Multi-class Classification Pipeline...")

    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plotRenderer = PlotRenderer()
    cplots = cp()

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df, report = dl.read_dataset("adultcensusincome.csv", optimize=False, handle_unnamed="drop", return_report=True)

    df_info = dfh.get_dataframe_info_str(df)

    df_usa = df[df['native.country'] == 'United-States']
    # ---------------------------------------------------
    # ✅ MULTI-CLASS TARGET
    # ---------------------------------------------------
    # Instead of binary income → use multi-class target
    # Example: education levels
    target_col = "education"   # ✅ multi-class

    # ---------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------
    imputer = CustomImputer(
        num_strategy="mode",
        groupby_cols=["native.country", "workclass", "occupation"]
    )

    outlier = OutlierHandler(method="iqr", factor=1.5)

    # ---------------------------------------------------
    # INIT MODEL UTILITY
    # ---------------------------------------------------
    cm = cmu(
        df_usa,
        target_col=target_col,
        imputer=imputer,
        outlier_handler=outlier
    )

    cm.prepare_data()

    # ---------------------------------------------------
    # RUN MODELS
    # ---------------------------------------------------
    results_df = cm.run_all_models()

    # ---------------------------------------------------
    # ✅ OPTIONAL TUNING
    # ---------------------------------------------------
    # cm.tune_model(
    #     "DecisionTreeClassifier",
    #     max_depth=[5, 10, 15]
    # )

    # ---------------------------------------------------
    # ✅ MODEL COMPARISON
    # ---------------------------------------------------
    ranked = cm.rank_models(metric="f1")
    best_model = cm.get_best_model(metric="f1")

    results_df = cm.get_results_df()

    # ---------------------------------------------------
    # ✅ VISUALIZATION
    # ---------------------------------------------------
    content.append(builder.grid([
        builder.card("Data Info", builder.render_pre(df_info)),
        builder.card("Model Results", builder.render_dataframe(results_df)),
        builder.card("Best Model", builder.render_dict(best_model)),
    ]))

    content.append(builder.chart_grid([
        plotRenderer.plot_to_card(cplots.plot_bar(results_df, metric="f1"), "F1 Score Comparison"),
        plotRenderer.plot_to_card(cplots.plot_multi_metrics(results_df), "Multi-Metric Comparison"),
        plotRenderer.plot_to_card(cplots.plot_roc_all_models(cm.results), "ROC Curve Comparison"),
    ]))

    # ---------------------------------------------------
    # REPORT
    # ---------------------------------------------------
    html_doc = builder.build_page(
        "Multi-Class Classification Report",
        "\n".join(content)
    )

    output_path = ru.save_html_report(
        __file__,
        "ml_multiclass_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )

    print(f"Wrote report to: {output_path}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
