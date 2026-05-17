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
    print("Running ml linear regression pipeline report...")
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
ml_results = ml.train_all(imputer=imputer, outlier_handler=outlier)
ml_kfold_results = ml.train_all(imputer=imputer, k_fold=5, outlier_handler=outlier)
mlplot = mpv(ml_results)
mlplot_kfold = mpv(ml_kfold_results)

# use for the large dataset
content.append(builder.full_width_card("Original Marketing Data",
                                       builder.render_dataframe_collapsible(df, initial_rows=15)))
content.append(
    builder.grid([
        builder.card("Dataframe Information:", builder.render_pre(df_info)),
        builder.card("Machine learning report:", builder.render_dict(ml_results)),
    ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(mlplot.plot_model_comparison(), " Liner Regression Model Performances"),
    # plotRenderer.plot_to_card(mlplot_kfold.plot_model_comparison(), " Liner Regression Model Performances with K-fold=5"),
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

if __name__ == "__main__":
    main()
