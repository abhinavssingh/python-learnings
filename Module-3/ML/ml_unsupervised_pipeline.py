import time

import pandas as pd

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.UnsupervisedModelUtility import UnsupervisedModelUtility
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.core.VisualizerEngine import VisualizerEngine
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running Unsupervised ML Pipeline Report...")
    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plotRenderer = PlotRenderer()

    # ========================================================
    # LOAD DATA
    # ========================================================
    df, report = dl.read_dataset(
        "adultcensusincome.csv",
        optimize=False,
        handle_unnamed="drop",
        return_report=True
    )

    df_info = dfh.get_dataframe_info_str(df)

    # ========================================================
    # INIT UTILITY
    # ========================================================
    imputer = CustomImputer(num_strategy="median")
    outlier = OutlierHandler(method="iqr", factor=1.5)

    X = df.copy().drop(columns=["income", "native.country"])

    um = UnsupervisedModelUtility(
        X=X,
        imputer=imputer,
        outlier_handler=outlier
    )

    # ========================================================
    # MODEL EXECUTION
    # ========================================================
    um.prepare_data()

    um.run_experiment("KMeans")
    um.run_experiment("DBSCAN")

    # ✅ Add tuned runs so result_df contains tuning rows for plot consumers
    # um.tune_model("KMeans", n_clusters=[2, 3, 4, 5])
    # um.tune_model("DBSCAN", eps=[0.3, 0.5, 0.7], min_samples=[5, 10])

    best_model = um.get_best_model(metric="silhouette_score")
    if best_model is None:
        raise ValueError("No valid unsupervised model found for saving/validation")

    um.save_model(best_model["experiment"], "saved_models/unsupervised/best_model")

    results_df = um.get_results_df()

    # ========================================================
    # ✅ VISUAL ENGINE
    # ========================================================
    plot_data = um.get_plot_data()
    viz = VisualizerEngine(
        um.results,
        plot_data
    )

    dashboard = viz.render_all()

    # ========================================================
    # VALIDATION
    # ========================================================
    validation_results = um.validate_inference_pipeline(best_model["experiment"], "saved_models/unsupervised/best_model")

    # ========================================================
    # REPORT CONTENT
    # ========================================================
    content.append(builder.grid([
        builder.card("Data Info", builder.render_pre(df_info)),
        builder.card("Processed Data", builder.render_dataframe(pd.DataFrame(plot_data["X_processed"]).head())),
        builder.card("Unsupervised Results", builder.render_dataframe(results_df)),
        builder.card("Validation Results", builder.render_dict(validation_results))
    ]))

    # ========================================================
    # ✅ VISUALIZATION (NOW FULLY UNIFIED ✅)
    # ========================================================
    content.append(builder.chart_grid([

        # =====================================================
        # ✅ GENERIC MODEL EVALUATION
        # =====================================================
        plotRenderer.plot_to_card(dashboard["comparison"], "Model Comparison"),
        plotRenderer.plot_to_card(dashboard["ranking"], "Model Ranking"),
        plotRenderer.plot_to_card(dashboard["best_model"], "Best Model"),
        plotRenderer.plot_to_card(dashboard["distribution"], "Metric Distribution"),

        # =====================================================
        # ✅ TASK-SPECIFIC VISUALS (ORDERED + CLEAN ✅)
        # =====================================================

        # ✅ Task-specific (Unsupervised plots)
        *[
            plotRenderer.plot_to_card(fig, title)
            for title, fig in dashboard["task_specific"].items()
        ],

    ]))

    # ========================================================
    # REPORT
    # ========================================================
    html_doc = builder.build_page(
        "Unsupervised ML Pipeline Report",
        "\n".join(content)
    )

    output_path = ru.save_html_report(
        __file__,
        "unsupervised_ml_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )

    print(f"Wrote report to: {output_path}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
