import time

import plotly.express as px

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.machinelearning.facade.ClassificationModelUtility import ClassificationModelUtility as cmu
from lib.utility.machinelearning.pipeline.CustomImputer import CustomImputer
from lib.utility.machinelearning.pipeline.OutlierHandler import OutlierHandler
from lib.utility.machinelearning.visualization.generic.ClassificationPlots import ClassificationPlots as cp
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running ml logistic regression pipeline report...")
    # ...


# Start the timer
start_time = time.perf_counter()

# initialization and set variable
content = []
dashboard = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()
cplots = cp()


df, report = dl.read_dataset("adultcensusincome.csv", optimize=False, handle_unnamed="drop", return_report=True)
df_info = dfh.get_dataframe_info_str(df)

# check for missing values represented as "?"
missing_count = df.isin(["?"]).sum().loc[lambda x: x > 0]

unique_values = df.select_dtypes(include=["str"]).nunique().to_dict()

# display top 10 countries and group the rest as "Others"
top_n = 10
country_counts = df['native.country'].value_counts()
top_countries = country_counts[:top_n]
others = country_counts[top_n:].sum()
top_countries['Others'] = others

# Create DataFrame
plot_df = top_countries.reset_index()
plot_df.columns = ['Country', 'Count']

# Add percentage column
total = plot_df['Count'].sum()
plot_df['Percentage'] = (plot_df['Count'] / total * 100).round(2)
plot_df['Percentage_str'] = plot_df['Percentage']

numeric_df = df.select_dtypes(include=['number'])
numeric_corr = numeric_df.corr()

univariate_pre = """
Univariate analysis is the simplest form of statistical data
analysis,used to examine a dataset containing only one
variable at a time. It does not look for relationships
or causes; rather, its primary purpose is to describe
the data, summarize its distribution,and identify
patterns like outliers.
"""

# univariate plots
country_bar_fig = px.bar(plot_df, x='Country', y='Count', labels={"Count": "Count", "Country": "Country"},
                         title="Top 10 Countries + Others", hover_data={"Count": True, "Percentage": True})

country_bar_fig.update_traces(text=plot_df['Percentage'].astype(str) + '%', textposition='outside')

hist_age_fig = px.histogram(df, x="age", marginal="box", opacity=0.7, barmode="overlay",
                            title="Histogram Box Graph with Outliers", hover_data=df.columns)

hist_income_fig = px.histogram(df, x="income", opacity=0.7, barmode="overlay",
                               title="Histogram Box Graph with Outliers", hover_data=df.columns)

hist_education_fig = px.histogram(df, x="education", marginal="box", opacity=0.7, barmode="overlay",
                                  title="Histogram Box Graph with Outliers", hover_data=df.columns)

hist_education_num_fig = px.histogram(df, x="education.num", marginal="box", opacity=0.7, barmode="overlay",
                                      labels={"education.num": "Education Number"}, title="Histogram Box Graph with Outliers", hover_data=df.columns)

marrital_pie_fig = px.pie(df, names="marital.status", title="Marital Status Distribution")

# bivariate plots
age_income_hist_fig = px.histogram(df, x="age", color="income", barmode="group", title="Income vs Age")

education_income_hist_fig = px.histogram(df, x="education", color="income", barmode="group", title="Income vs Education")

marital_status_hist_fig = px.histogram(df, x="marital.status", color="income", barmode="group",
                                       labels={"marital.status": "Marital Status"}, title="Income vs Marital Status")

sex_income_hist_fig = px.histogram(df, x="sex", color="income", barmode="group", title="Income vs Sex")

nuemric_corr_fig = px.imshow(numeric_corr, text_auto=True, color_continuous_scale="RdBu", title="Correlation Heatmap")

# use for the large dataset
content.append(builder.full_width_card("Original Marketing Data",
                                       builder.render_dataframe_collapsible(df, initial_rows=15)))

# ===================================================================
# Machine Learning Pipeline
# ===================================================================
# filter for USA only to speed up the process because dataset is imbalanced and has many rows,
# you can remove this filter to run on the full dataset but it will take much longer time to execute
df_usa = df[df['native.country'] == 'United-States']
# initializing imputer and outlier handler with custom settings
imputer = CustomImputer(num_strategy="mode", groupby_cols=["native.country", "workclass", "occupation"])
outlier = OutlierHandler(method="iqr", factor=1.5)

# initialize ClassificationModelUtility with the dataframe and target column
cm = cmu(df_usa, target_col="income", imputer=imputer, outlier_handler=outlier)

# prepare data (handle missing values, outliers, etc.)
cm.prepare_data()

# run all models with default settings
ml_results = cm.run_all_models()


# ✅ tuning

# cm.tune_model("LogisticRegression", C=[0.1, 1, 10], solver=["lbfgs"])
# cm.tune_model("DecisionTreeClassifier", max_depth=[5, 10], min_samples_leaf=[2, 5])
# cm.tune_model("RandomForestClassifier", n_estimators=[10, 20, 50], max_depth=[5, 10])

# param_configs = {
#     "LogisticRegression": {
#         "model__C": [0.1, 1, 10]
#     },
#     "KNNClassifier": {
#         "model__n_neighbors": [3, 5, 7],
#         "model__weights": ["uniform", "distance"]
#     }
# }

# cm.tune_all_models(param_configs, search_type="grid", cv=5, scoring="f1")

models_ranked = cm.rank_models(metric="f1")
models_comparison = cm.compare_models()

results_df = cm.get_results_df()
artifacts_df = cm.get_artifacts_df()
# ===================================================================
# RESULTS SECTION 1: Basic Info & Train All Results
# ===================================================================
content.append(builder.grid([
    builder.card("Dataframe Description:", builder.render_dict(df.describe().to_dict())),
    builder.card("Dataframe Information:", builder.render_pre(df_info)),
    builder.card("Missing Values (as '?'):", builder.render_series(missing_count)),
    builder.card("Unique Values Count per Column:", builder.render_dict(unique_values)),
    builder.card("Univariate Analysis:", builder.render_pre(univariate_pre)),
    builder.card("Train All Classification Models:", builder.render_dict(ml_results.to_dict())),
    builder.card("All Classification Models Results (Flat)", builder.render_dataframe(results_df)),
    # builder.card("Artifacts for all classification models:", builder.render_dataframe(artifacts_df)),
    builder.card("Confusion Matrix for all classification models:", builder.render_dict(cm.get_all_confusion_matrices()))

]))

# ===================================================================
# VISUALIZATION SECTION: Model Comparisons
# ===================================================================

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(country_bar_fig, "Top 10 Countries + Others"),
    plotRenderer.plot_to_card(hist_age_fig, " Census Age Distribution"),
    plotRenderer.plot_to_card(hist_income_fig, "Census Income Distribution"),
    plotRenderer.plot_to_card(hist_education_fig, "Census Education Distribution"),
    plotRenderer.plot_to_card(hist_education_num_fig, "Census Education Number Distribution"),
    plotRenderer.plot_to_card(marrital_pie_fig, "Marital Status Distribution"),
    plotRenderer.plot_to_card(age_income_hist_fig, "Income vs Age"),
    plotRenderer.plot_to_card(education_income_hist_fig, "Income vs Education"),
    plotRenderer.plot_to_card(marital_status_hist_fig, "Income vs Marital Status"),
    plotRenderer.plot_to_card(sex_income_hist_fig, "Income vs Sex"),
    plotRenderer.plot_to_card(nuemric_corr_fig, "Numeric Correlation Heatmap"),
    plotRenderer.plot_to_card(cplots.plot_bar(results_df, metric="accuracy"), "Preprocessing Impact"),
    plotRenderer.plot_to_card(cplots.plot_bar(results_df, metric="f1"), "Train vs KFold"),
    plotRenderer.plot_to_card(cplots.plot_best_model(results_df, metric="f1"), "Best Model (Annotated)"),
    plotRenderer.plot_to_card(cplots.plot_multi_metrics(results_df, metrics=["accuracy", "f1", "precision", "recall", "roc_auc"]), "Multi-Metric Comparison"),
    plotRenderer.plot_to_card(cplots.plot_roc_all_models(cm.results), "ROC Curves for All Models"),
]))

html_doc = builder.build_page(
    "ML Logistic Regression Pipeline Report",
    "\n".join(content))

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "ml_logistic_regression_pipeline_report.html",   # file name
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
