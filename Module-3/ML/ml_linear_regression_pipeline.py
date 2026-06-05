import time

import numpy as np
import pandas as pd

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.LinearModelUtility import LinearModelUtility as lmu
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.advanced.ComparisonPlots import ComparisonPlots
from lib.utility.machinelearning.visualization.advanced.HyperparameterPlots import HyperparameterPlots
from lib.utility.machinelearning.visualization.advanced.OptimizationPlots import OptimizationPlots
from lib.utility.machinelearning.visualization.generic.ModelPerformanceVisualizer import ModelPerformanceVisualizer as mpv
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running ml linear regression pipeline report...")
    # ...


# Start the timer
start_time = time.perf_counter()

# initialization and set variable
content = []
dashboard = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()
mlplot = mpv()
cmp = ComparisonPlots()
opt = OptimizationPlots()
hp = HyperparameterPlots()


df, report = dl.read_dataset("marketing_data.csv", optimize=False, handle_unnamed="drop", return_report=True)

# Ensure Income is numeric (remove $ and commas if needed)
df['Income'] = df['Income'].replace('[\\$,]', '', regex=True).astype(float)

# convert Formatted Date column to date time
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# Select columns that contain 'Mnt' and calculate row-wise sum
Total_Mnt = df.loc[:, df.columns.str.contains('Mnt')].sum(axis=1)

df = dfh.insert_column_after(
    df, after_col="MntGoldProds", new_col="TotalSpend", values=Total_Mnt, inplace=True)

df_info = dfh.get_dataframe_info_str(df)

# initializing imputer and outlier handler with custom settings
imputer = CustomImputer(num_strategy="mean", groupby_cols=["Education", "Marital_Status"])
outlier = OutlierHandler(method="iqr", factor=1.5)

# initialize LinearModelUtility with the dataframe and target column
lm = lmu(df, target_col="TotalSpend", imputer=imputer, outlier_handler=outlier)

# prepare data (handle missing values, outliers, etc.)
lm.prepare_data()

# run all models with default settings
ml_results = lm.run_all_models()

# run K-Fold cross-validation for all models
ml_kfold_results = lm.run_experiment(model_name="LinearRegression", k_fold=5)

# run selected models with different parameters
configs = [
    {"model_name": "Ridge", "k_fold": 5},
    {"model_name": "Lasso", "k_fold": 5},
    {"model_name": "ElasticNet", "k_fold": 10},
    {"model_name": "Ridge", "imputer": imputer},
    {"model_name": "Ridge", "outlier_handler": outlier},
]
ml_selected_results = lm.run_experiments(configs)

# define parameter grid for Ridge regressionand perform grid search
param_grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
ridge_grid_result = lm.grid_search_cv(model_name="Ridge", param_grid=param_grid)

# tuned both grid search and random search for Ridge and ElasticNet respectively
ridge_tuned_result = lm.tune_model(model_name="Ridge", param_grid=param_grid, search_type="grid",)

param_dist = {
    "model__alpha": np.linspace(0.01, 1, 20),
    "model__l1_ratio": np.linspace(0.1, 0.9, 10)
}
elasticnet_tuned_result = lm.tune_model(model_name="ElasticNet", param_grid=param_dist, search_type="random", n_iter=15)


ranking = lm.rank_models("R2")
best_model = lm.get_best_model("R2")
comparison = lm.compare_models()
results_df = lm.get_results_df()

improvement_df = lm.compare_models()


# use for the large dataset
content.append(builder.full_width_card("Original Marketing Data",
                                       builder.render_dataframe_collapsible(df, initial_rows=15)))

# ===================================================================
# RESULTS SECTION 1: Basic Info & Train All Results
# ===================================================================
content.append(builder.grid([
    builder.card("Dataframe Information:", builder.render_pre(df_info)),
    builder.card("Train All (all 5 models)", builder.render_dict(ml_results.to_dict())),
    builder.card("Train LinearRegression models with K-Fold (k=5):", builder.render_dict(ml_kfold_results)),
    builder.card("Train Selected models with different parameters:", builder.render_dict(ml_selected_results.to_dict())),
    builder.card("Ridge Grid Search Result:", builder.render_dict(ridge_grid_result)),
    builder.card("Ridge Tuned Result (Grid Search):", builder.render_dataframe(pd.DataFrame(ridge_tuned_result))),
    builder.card("ElasticNet Tuned Result (Random Search):", builder.render_dataframe(pd.DataFrame(elasticnet_tuned_result))),
    builder.card("Model Ranking (R2)", builder.render_dataframe(ranking)),
    builder.card("Best Model", builder.render_dict(best_model)),
    builder.card("Model Comparison Summary", builder.render_dataframe(comparison)),
    builder.card("All Results (Flat)", builder.render_dataframe(results_df)),
]))

# ===================================================================
# VISUALIZATION SECTION: Model Comparisons
# ===================================================================

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(cmp.plot_preprocessing_impact(results_df, "MSE", "R2"), "Preprocessing Impact"),
    plotRenderer.plot_to_card(cmp.plot_mode_comparison(results_df, metric="R2"), "Train vs KFold"),
    plotRenderer.plot_to_card(cmp.plot_best_model_highlight(results_df, metric="R2"), "Best Model (Annotated)"),
    plotRenderer.plot_to_card(opt.plot_optimization_animation(results_df, "MSE", "R2"), "Optimization Animation"),
    plotRenderer.plot_to_card(hp.plot_3d_surface(results_df, "param_model__alpha", "param_model__l1_ratio"),
                              "3D Hyperparameter Surface"),
    plotRenderer.plot_to_card(opt.plot_search_animation(results_df, model="Ridge", mode="gridsearch"),
                              "Grid Search Animation"),

]))

html_doc = builder.build_page(
    "ML Linear Regression Pipeline Report",
    "\n".join(content))

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "ml_linear_regression_pipeline_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)

print(f"Wrote report to: {output_path}")

# End the timer
end_time = time.perf_counter()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.6f} seconds")

if __name__ == "__main__":
    main()
