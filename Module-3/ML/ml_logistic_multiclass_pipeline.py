import time

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.ClassificationModelUtility import ClassificationModelUtility as cmu
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.core.VisualizerEngine import VisualizerEngine
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running Multi-class Classification Pipeline...")
    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plotRenderer = PlotRenderer()

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df, report = dl.read_dataset(
        "adultcensusincome.csv",
        optimize=False,
        handle_unnamed="drop",
        return_report=True
    )

    df_info = dfh.get_dataframe_info_str(df)

    # ---------------------------------------------------
    # ✅ MULTI-CLASS TARGET
    # ---------------------------------------------------
    target_col = "education"

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
        df,
        target_col=target_col,
        imputer=imputer,
        outlier_handler=outlier
    )

    cm.prepare_data()

    # ---------------------------------------------------
    # RUN MODELS
    # ---------------------------------------------------
    cm.run_all_models()

    # ---------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------
    best_model = cm.get_best_model(metric="f1_macro")

    results_df = cm.get_results_df()
    plot_data = cm.get_plot_data()  # ✅ get_plot_data to extract necessary data for visualizations

    # ✅ IMPORTANT: pass BOTH results + artifacts
    viz = VisualizerEngine(
        cm.results,
        plot_data
    )

    dashboard = viz.render_all()

    # ---------------------------------------------------
    # REPORT CONTENT
    # ---------------------------------------------------
    content.append(builder.grid([
        builder.card("Data Info", builder.render_pre(df_info)),
        builder.card("Model Results", builder.render_dataframe(results_df)),
        builder.card("Best Model", builder.render_dict(best_model)),
    ]))

    # ---------------------------------------------------
    # ✅ VISUALIZATION (NEW ✅)
    # ---------------------------------------------------
    content.append(
        builder.chart_grid([

            # ✅ Generic plots
            plotRenderer.plot_to_card(dashboard["comparison"], "Model Comparison"),
            plotRenderer.plot_to_card(dashboard["ranking"], "Model Ranking"),
            plotRenderer.plot_to_card(dashboard["best_model"], "Best Model"),
            plotRenderer.plot_to_card(dashboard["distribution"], "Metric Distribution"),

            # ✅ Task-specific (classification)
            *[
                plotRenderer.plot_to_card(fig, title)
                for title, fig in dashboard["task_specific"].items()
            ]

        ])
    )

    # ---------------------------------------------------
    # REPORT OUTPUT
    # ---------------------------------------------------
    html_doc = builder.build_page("Multi-Class Classification Report", "\n".join(content))

    output_path = ru.save_html_report(__file__, "ml_multiclass_report.html", html_doc, subfolder="reports", open_in_browser=True)

    print(f"Wrote report to: {output_path}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
