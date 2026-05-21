import pandas as pd

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.CustomImputer import CustomImputer
from lib.utility.machinelearning.LinearModelUtility import LinearModelUtility as lmu
from lib.utility.machinelearning.ModelPerformanceVisualizer import ModelPerformanceVisualizer as mpv
from lib.utility.machinelearning.OutlierHandler import OutlierHandler
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running ml linear regression k-fold pipeline report...")
    # ...


# initialization and set variable
content = []
dashboard = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()

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

# initilaizaing machine learning pipeline
ml = lmu(df, target_col="TotalSpend")
imputer = CustomImputer(num_strategy="mean", groupby_cols=["Education", "Marital_Status"])
outlier = OutlierHandler(method="iqr", factor=1.5)
ml_kfold_results = ml.train_all(imputer=imputer, k_fold=5, outlier_handler=outlier)
mlplot_kfold = mpv(ml_kfold_results)

# use for the large dataset
content.append(builder.full_width_card("Original Marketing Data",
                                       builder.render_dataframe_collapsible(df, initial_rows=15)))
content.append(
    builder.grid([
        builder.card("Dataframe Information:", builder.render_pre(df_info)),
        builder.card("Machine learning report with K-fold:", builder.render_dict(ml_kfold_results)),
    ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(mlplot_kfold.plot_model_comparison(["Ridge_k5", "Lasso_k5"]), " Ridge vs Lasso Model Performances with K-fold=5"),
    plotRenderer.plot_to_card(mlplot_kfold.plot_all_model_comparison(), " All Liner Regression Model Performances with K-fold=5"),
    plotRenderer.plot_to_card(mlplot_kfold.plot_total_error_all(mode="squared"), " Total Error for all Models with K-fold=5"),
    plotRenderer.plot_to_card(mlplot_kfold.plot_actual_vs_predicted("LinearRegression_k5"), "LinearRegression Actual vs Predicted Performance with K-fold=5"),
    plotRenderer.plot_to_card(mlplot_kfold.plot_actual_vs_predicted("SGDRegressor_k5"), "SGDRegressor Actual vs Predicted Performance with K-fold=5"),
    plotRenderer.plot_to_card(mlplot_kfold.plot_actual_vs_predicted("Ridge_k5"), "Ridge Actual vs Predicted Performance with K-fold=5"),
    plotRenderer.plot_to_card(mlplot_kfold.plot_actual_vs_predicted("Lasso_k5"), "Lasso Actual vs Predicted Performance with K-fold=5"),
    plotRenderer.plot_to_card(mlplot_kfold.plot_actual_vs_predicted("ElasticNet_k5"), "ElasticNet Actual vs Predicted Performance with K-fold=5"),
]))

html_doc = builder.build_page(
    "ML Linear Regression K-FoldPipeline Report",
    "\n".join(content))

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "ml_linear_regression__kfold_pipeline_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)

print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
