import time

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.machinelearning.facade.ClassificationModelUtility import ClassificationModelUtility as cmu
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.core.VisualizerEngine import VisualizerEngine
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running Multi-label Classification Pipeline...")
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

    # ---------------------------------------------------
    # ✅ CREATE MULTI-LABEL TARGET
    # ---------------------------------------------------
    df["labels"] = df.apply(
        lambda x: list(filter(None, [

            "high_income" if x["income"] == ">50K" else None,
            "high_edu" if x["education.num"] >= 10 else None,
            "low_edu" if x["education.num"] < 6 else None,
            "senior" if x["age"] > 50 else None,
            "young" if x["age"] < 30 else None,
            "private_emp" if x["workclass"] == "Private" else None,

        ])),
        axis=1
    )

    mlb = MultiLabelBinarizer()

    y = pd.DataFrame(
        mlb.fit_transform(df["labels"]),
        columns=mlb.classes_
    ).reset_index(drop=True)

    # ✅ Remove invalid labels
    y = y.loc[:, y.nunique() > 1]

    # ✅ FEATURES
    X = df.drop(columns=["labels", "income"]).reset_index(drop=True)

    df_ml = pd.concat([X, y], axis=1)
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
    # INIT UTILITY
    # ---------------------------------------------------
    cm = cmu(
        df_ml,
        target_col=target_cols,
        imputer=imputer,
        outlier_handler=outlier
    )

    cm.prepare_data()

    # ---------------------------------------------------
    # RUN MODELS
    # ---------------------------------------------------
    cm.run_all_models()

    results_df = cm.get_results_df()

    # ---------------------------------------------------
    # ✅ MODEL ANALYSIS
    # ---------------------------------------------------
    best_model = cm.get_best_model(metric="f1_weighted")
    plot_data = cm.get_plot_data()  # ✅ get_plot_data to extract necessary data for visualizations

    # ✅ Visualizer Engine (CRITICAL ✅)
    viz = VisualizerEngine(
        cm.results,
        plot_data
    )

    dashboard = viz.render_all()

    # ---------------------------------------------------
    # REPORT CONTENT
    # ---------------------------------------------------
    content.append(builder.grid([
        builder.card("Results", builder.render_dataframe(results_df)),
        builder.card("Best Model", builder.render_dict(best_model)),
    ]))

    # ---------------------------------------------------
    # ✅ VISUALIZATION (UPDATED ✅)
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
