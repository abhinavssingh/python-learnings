import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.LinearModelUtility import LinearModelUtility
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.advanced.HyperparameterPlots import HyperparameterPlots
from lib.utility.machinelearning.visualization.advanced.OptimizationPlots import OptimizationPlots
from lib.utility.machinelearning.visualization.core.VisualizerEngine import VisualizerEngine
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    print("Running ml linear regression pipeline report...")
    start_time = time.perf_counter()

    content = []
    builder = HtmlBuilder()
    plotRenderer = PlotRenderer()

    # ✅ Visualization components
    viz = None
    opt = OptimizationPlots()
    hp = HyperparameterPlots()

    # --------------------------------------------------
    # ✅ LOAD DATA
    # --------------------------------------------------
    df, report = dl.read_dataset(
        "marketing_data.csv",
        optimize=False,
        handle_unnamed="drop",
        return_report=True
    )

    df['Income'] = df['Income'].replace('[\\$,]', '', regex=True).astype(float)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

    Total_Mnt = df.loc[:, df.columns.str.contains('Mnt')].sum(axis=1)

    df = dfh.insert_column_after(
        df,
        after_col="MntGoldProds",
        new_col="TotalSpend",
        values=Total_Mnt,
        inplace=True
    )

    df_info = dfh.get_dataframe_info_str(df)

    # --------------------------------------------------
    # ✅ PIPELINE SETUP
    # --------------------------------------------------
    # ✅ CONFIG
    imputer = CustomImputer(num_strategy="mean", groupby_cols=["Education", "Marital_Status"])
    outlier = OutlierHandler(method="iqr", factor=1.5)

    # ✅ SPLIT data
    X = df.drop("TotalSpend", axis=1)
    y = df["TotalSpend"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ PASS SPLIT DATA
    lm = LinearModelUtility(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, imputer=imputer, outlier_handler=outlier)

    lm.prepare_data()

    # --------------------------------------------------
    # ✅ MODEL EXECUTION
    # --------------------------------------------------
    ml_results = lm.run_all_models()
    ml_kfold_results = lm.run_experiment(model_name="LinearRegression", k_fold=5)

    configs = [
        {"model_name": "Ridge", "k_fold": 5},
        {"model_name": "Lasso", "k_fold": 5},
        {"model_name": "ElasticNet", "k_fold": 10},
        {"model_name": "Ridge", "imputer": imputer},
        {"model_name": "Ridge", "outlier_handler": outlier},
    ]

    ml_selected_results = lm.run_experiments(configs)

    # ✅ tuning
    param_grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    ridge_grid_result = lm.grid_search_cv(model_name="Ridge", param_grid=param_grid)

    ridge_tuned_result = lm.tune_model(
        model_name="Ridge",
        param_grid=param_grid,
        search_type="grid"
    )

    param_dist = {
        "model__alpha": np.linspace(0.01, 1, 20),
        "model__l1_ratio": np.linspace(0.1, 0.9, 10)
    }

    elasticnet_tuned_result = lm.tune_model(
        model_name="ElasticNet",
        param_grid=param_dist,
        search_type="random",
        n_iter=15
    )

    # --------------------------------------------------
    # ✅ EVALUATION
    # --------------------------------------------------
    ranking = lm.rank_models("R2")
    best_model = lm.get_best_model(metric="R2")
    lm.save_model(best_model["experiment"], "saved_models/regression/best_model")
    comparison = lm.compare_models()
    results_df = lm.get_results_df()

    # ✅ Initialize visualizer AFTER results
    viz = VisualizerEngine(lm.results, lm.artifacts if hasattr(lm, "artifacts") else {})

    dashboard = viz.render_all()

    # ✅ VALIDATE
    validation = lm.validate_inference_pipeline(exp_id=best_model["experiment"], model_path="saved_models/regression/best_model")

    # --------------------------------------------------
    # ✅ REPORT SECTION 1 (DATA + RESULTS)
    # --------------------------------------------------
    content.append(
        builder.full_width_card(
            "Original Marketing Data",
            builder.render_dataframe_collapsible(df, initial_rows=15)
        )
    )

    content.append(builder.grid([
        builder.card("Dataframe Info", builder.render_pre(df_info)),
        builder.card("All Models", builder.render_dataframe(ml_results)),
        builder.card("KFold Results", builder.render_dict(ml_kfold_results)),
        builder.card("Selected Models", builder.render_dict(ml_selected_results.to_dict())),
        builder.card("Grid Search", builder.render_dict(ridge_grid_result)),
        builder.card("Ridge Tuned", builder.render_dataframe(pd.DataFrame(ridge_tuned_result))),
        builder.card("ElasticNet Tuned", builder.render_dataframe(pd.DataFrame(elasticnet_tuned_result))),
        builder.card("Ranking", builder.render_dataframe(ranking)),
        builder.card("Best Model", builder.render_dict(best_model)),
        builder.card("Comparison", builder.render_dataframe(comparison)),
        builder.card("All Results", builder.render_dataframe(results_df)),
        builder.card("Inference Validation", builder.render_dict(validation))
    ]))

    # --------------------------------------------------
    # ✅ VISUALIZATION SECTION (NEW ✅)
    # --------------------------------------------------
    content.append(builder.chart_grid([

        plotRenderer.plot_to_card(dashboard["comparison"], "Model Comparison"),
        plotRenderer.plot_to_card(dashboard["ranking"], "Model Ranking"),
        plotRenderer.plot_to_card(dashboard["best_model"], "Best Model"),
        plotRenderer.plot_to_card(dashboard["distribution"], "Metric Distribution"),

        # ✅ Example task-specific charts (regression)
        *[
            plotRenderer.plot_to_card(fig, title)
            for title, fig in dashboard["task_specific"].items()
        ],

        # ✅ Advanced plots (kept)
        plotRenderer.plot_to_card(
            opt.plot_optimization_animation(results_df, "MSE", "R2"),
            "Optimization Animation"
        ),

        plotRenderer.plot_to_card(
            hp.plot_3d_surface(results_df, "param_model__alpha", "param_model__l1_ratio"),
            "Hyperparameter Surface"
        ),
    ]))

    # --------------------------------------------------
    # ✅ FINAL HTML
    # --------------------------------------------------
    html_doc = builder.build_page(
        "ML Linear Regression Pipeline Report",
        "\n".join(content)
    )

    output_path = ru.save_html_report(
        __file__,
        "ml_linear_regression_pipeline_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )

    print(f"Wrote report to: {output_path}")

    # --------------------------------------------------
    # ✅ EXECUTION TIME
    # --------------------------------------------------
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
