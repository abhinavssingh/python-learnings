import numpy as np
import pandas as pd

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.plot.plotutility import PlotUtility as pu
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Capstone-1 report...")
    # ...

# Build full column-wise page


builder = HtmlBuilder()

df = dl.read_dataset("NSMES1988.csv", handle_unnamed="drop")
df_copy = df.copy()
# float16 is fine for storage but not supported for Index / binning operations
#  float16 is best for ML tensors, not analytics
df_copy["age"] = np.floor(df_copy["age"] * 10).astype("float32")
df_copy["income"] = np.floor(df_copy["income"] * 10000).astype("float32")

df.rename(columns={
    "age": "Age in years (divided by 10)",
    "income": "Family income in USD 10000"
}, inplace=True)

# convert column name to title case
df.columns = [col.title() for col in df.columns]

df_numeric_uint = df.select_dtypes(include=["uint8", "uint16"])
df_numeric_float = df.select_dtypes(include=["float16", "float32"])
# df.to_json("NSMES1988.json", orient="records", lines=True)  # Save as JSON for future use

# operations on modified dataframe
df_copy.columns = [col.title() for col in df_copy.columns]

# 2. Define the bin edges and corresponding labels
age_bins = [0, 2, 4, 13, 20, 60, 75, 100, np.inf]  # np.inf for the upper limit
age_labels = ['Infant', 'Toddler', 'Kid', 'Teen',
              'Adult', 'Senior', 'Super Senior', 'Ultra Senior']

income_bins = [0, 40000, 80000, float('inf')]  # Bins up to infinity
income_labels = ['Low', 'Medium', 'High']

# 3. Use pd.cut() to create the new 'AgeGroup' column
# The default 'right=True' means the bins are (lower, upper], so (0, 2]
# To include the lower bound, you can use 'right=False', resulting in [0, 2)
age_category = pd.cut(df_copy['Age'], bins=age_bins, labels=age_labels)
income_category = pd.cut(
    df_copy['Income'], bins=income_bins, labels=income_labels)

# If your insert_column_after() implementation does true in‑place
# mutation, then this assignment can silently replace your DataFrame with
# None or with an unchanged object depending on the implementation.
# so use inplace = false
df_copy = dfh.insert_column_after(
    df_copy, after_col="Age", new_col="Age Category", values=age_category, inplace=False)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Income", new_col="Currency", values="USD", inplace=False)
df_copy = dfh.insert_column_after(
    df_copy,
    after_col="Currency",
    new_col="Income Category",
    values=income_category,
    inplace=False)

age_income_df = df_copy.groupby(
    ["Age Category", "Income Category"]).size().reset_index(name='Count')
health_age_df = df_copy.groupby(
    ["Health", "Age Category"]).size().reset_index(name='Count')
age_gender_df = df_copy.groupby(
    ["Age Category", "Gender"]).size().reset_index(name='Count')

# Create a Pivot Data

pivot_age_income = age_income_df.pivot(
    index="Income Category",
    columns="Age Category",
    values="Count"
)


pivot_health_age = health_age_df.pivot(
    index="Health",
    columns="Age Category",
    values="Count"
)

pivot_age_gender = age_gender_df.pivot(
    index="Gender",
    columns="Age Category",
    values="Count"
)


plot_age_income = pu.plot_heatmap(
    age_income_df,
    index="Income Category",
    columns="Age Category",
    backend="plotly",
    title="Age vs Income Category",
)


plot_health_age = pu.plot_heatmap(
    health_age_df,
    index="Health",
    columns="Age Category",
    backend="plotly",
    title="Age vs Health",
)

plot_age_gender = pu.plot_heatmap(
    age_gender_df,
    index="Gender",
    columns="Age Category",
    backend="plotly",
    title="Age vs Gender",
)

content = []

# use for the large dataset
content.append(
    builder.full_width_card(
        "Capstone-1 Data– Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(
    builder.full_width_card(
        "Integer Dataframe", builder.render_dataframe_collapsible(df_numeric_uint))
)

content.append(
    builder.full_width_card(
        "Float Dataframe", builder.render_dataframe_collapsible(df_numeric_float))
)

content.append(
    builder.full_width_card(
        "Modified Dataframe", builder.render_dataframe_collapsible(df_copy))
)

# Get information about the DataFrame

df_info_str = dfh.get_dataframe_info_str(df)
plotRendrer = PlotRenderer()
content.append(
    builder.grid([
        builder.card("Information of the Capstone-1 Unmodified Dataframe is:",
                     builder.render_pre(df_info_str)),
        builder.card("Basic description of modified Dateframe:",
                     builder.render_dict(df_copy.describe().to_dict())),
        builder.card("Age vs Income Category Summary:",
                     builder.render_dataframe(age_income_df)),
        builder.card("Age vs Gender Summary:",
                     builder.render_dataframe(age_gender_df)),
        builder.card("Health vs Age Summary:",
                     builder.render_dataframe(health_age_df))
    ])
)

content.append(
    builder.chart_grid([
        plotRendrer.plot_to_card(
            plot_age_income, "Age vs Income"),
        plotRendrer.plot_to_card(
            plot_health_age, "Health vs Age",),
        plotRendrer.plot_to_card(
            plot_age_gender, "Age vs Gender"),
    ])
)

html_doc = builder.build_page(
    "Capstone-1 Exercise  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "capstone_1_exercise_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
