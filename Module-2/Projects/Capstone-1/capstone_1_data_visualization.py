import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Capstone-1 Data Visualization report...")
    # ...


# Build full column-wise page
builder = HtmlBuilder()
plotRenderer = PlotRenderer()
content = []

df = dl.read_dataset("NSMES1988.csv", handle_unnamed="drop")
# float16 is fine for storage but not supported for Index / binning operations
#  float16 is best for ML tensors, not analytics
df["age"] = np.floor(df["age"].mul(10, fill_value=1)).astype("float32")
df["income"] = np.floor(df["income"] * 10000).astype("float32")

# 2. Define the bin edges and corresponding labels
age_bins = [0, 2, 4, 13, 20, 60, 75, 100, np.inf]  # np.inf for the upper limit
age_labels = ['Infant', 'Toddler', 'Kid', 'Teen',
              'Adult', 'Senior', 'Super Senior', 'Ultra Senior']

income_bins = [0, 40000, 80000, float('inf')]  # Bins up to infinity
income_labels = ['Low', 'Medium', 'High']

# 3. Use pd.cut() to create the new 'AgeGroup' column
# The default 'right=True' means the bins are (lower, upper], so (0, 2]
# To include the lower bound, you can use 'right=False', resulting in [0, 2)
age_category = pd.cut(df['age'], bins=age_bins, labels=age_labels)
income_category = pd.cut(
    df['income'], bins=income_bins, labels=income_labels)

# If your insert_column_after() implementation does true in‑place
# mutation, then this assignment can silently replace your DataFrame with
# None or with an unchanged object depending on the implementation.
# so use inplace = false
df = dfh.insert_column_after(
    df, after_col="age", new_col="age category", values=age_category, inplace=False)
df = dfh.insert_column_after(
    df, after_col="income", new_col="currency", values="USD", inplace=False)
df = dfh.insert_column_after(
    df,
    after_col="currency",
    new_col="income category",
    values=income_category,
    inplace=False)

age_income_df = df.groupby(
    ["age category", "income category"]).size().reset_index(name='count')
health_age_df = df.groupby(
    ["health", "age category"]).size().reset_index(name='count')
age_gender_df = df.groupby(
    ["age category", "gender"]).size().reset_index(name='count')
health_gender_df = df.groupby(
    ["health", "gender"]).size().reset_index(name='count')
income_gender_df = df.groupby(
    ["income category", "gender"]).size().reset_index(name='count')
region_income_df = df.groupby(
    ["region", "income category"]).size().reset_index(name='count')
region_health_df = df.groupby(
    ["region", "health"]).size().reset_index(name='count')

category_df = df.groupby(
    ["region", "age category", "income category", "health", "gender"]).size().reset_index(name='count')


"""
below blocks create table chart plot but hang system in full width mode
table_data = age_income_df

fig = ff.create_table(table_data, height_constant=60)

trace1 = go.Scatter(x=age_income_df.columns, y=age_income_df.sort_values(by=['age category', 'income category'], ascending=[True, True]),
                    marker=dict(color='#0099ff'),
                    name='Goals',
                    xaxis='x2', yaxis='y2')

fig.add_traces([trace1])
# initialize xaxis2 and yaxis2
fig['layout']['xaxis2'] = {}
fig['layout']['yaxis2'] = {}

# Edit layout for subplots
fig.layout.xaxis.update({'domain': [0, .5]})
fig.layout.xaxis2.update({'domain': [0.6, 1.]})

# The graph's yaxis MUST BE anchored to the graph's xaxis
fig.layout.yaxis2.update({'anchor': 'x2'})
fig.layout.yaxis2.update({'title': 'Goals'})
"""

age_income_heatmap_df = (
    age_income_df
    .pivot(
        index="income category",   # y‑axis
        columns="age category",    # x‑axis
        values="count"             # cell value
    )
    .fillna(0)
)

age_income_fig = px.imshow(
    age_income_heatmap_df,
    text_auto=True,                 # show values in cells
    color_continuous_scale="RdPu",
    labels={
        "x": "age category",
        "y": "income category",
        "color": "count",
    },
    title="Age category vs Income Category"
)

region_income_heatmap_df = (
    region_income_df
    .pivot(
        index="income category",   # y‑axis
        columns="region",    # x‑axis
        values="count"             # cell value
    )
    .fillna(0)
)

region_income_fig = px.imshow(
    region_income_heatmap_df,
    text_auto=True,                 # show values in cells
    color_continuous_scale="BrBG",
    labels={
        "x": "region",
        "y": "income category",
        "color": "count",
    },
    title="Region vs Income Category"
)

health_age_heatmap_df = (
    health_age_df
    .pivot(
        index="health",   # y‑axis
        columns="age category",    # x‑axis
        values="count"             # cell value
    )
    .fillna(0)
)

health_age_fig = px.imshow(
    health_age_heatmap_df,
    text_auto=True,                 # show values in cells
    color_continuous_scale="Spectral",
    labels={
        "x": "age category",
        "y": "health",
        "color": "count",
    },
    title="Age Category vs Health"
)

region_health_heatmap_df = (
    region_health_df
    .pivot(
        index="health",   # y‑axis
        columns="region",    # x‑axis
        values="count"             # cell value
    )
    .fillna(0)
)

region_health_fig = px.imshow(
    region_health_heatmap_df,
    text_auto=True,                 # show values in cells
    color_continuous_scale="Oxy",
    labels={
        "x": "region",
        "y": "health",
        "color": "count",
    },
    title="Region vs Health"
)

health_gender_heatmap_df = (
    health_gender_df
    .pivot(
        index="gender",   # y‑axis
        columns="health",    # x‑axis
        values="count"             # cell value
    )
    .fillna(0)
)

health_gender_fig = px.imshow(
    health_gender_heatmap_df,
    text_auto=True,                 # show values in cells
    color_continuous_scale="YlGnBu",
    labels={
        "x": "health",
        "y": "gender",
        "color": "count",
    },
    title="Health vs Gender"
)

age_gender_heatmap_df = (
    age_gender_df
    .pivot(
        index="gender",   # y‑axis
        columns="age category",    # x‑axis
        values="count"             # cell value
    )
    .fillna(0)
)

age_gender_fig = px.imshow(
    age_gender_heatmap_df,
    text_auto=True,                 # show values in cells
    color_continuous_scale="RdGy",
    labels={
        "x": "age category",
        "y": "gender",
        "color": "count",
    },
    title="Age category vs Gender"
)

income_gender_heatmap_df = (
    income_gender_df
    .pivot(
        index="gender",   # y‑axis
        columns="income category",    # x‑axis
        values="count"             # cell value
    )
    .fillna(0)
)

income_gender_fig = px.imshow(
    income_gender_heatmap_df,
    text_auto=True,                 # show values in cells
    color_continuous_scale="sunset",
    labels={
        "x": "income category",
        "y": "gender",
        "color": "count",
    },
    title="Income Category vs Gender"
)

scatter_fig = px.scatter(df, x="age", y="visits", title="Scatter Plot to show density of visits vs Age", color="region",
                         size='age', hover_data=['gender'])

dims_1 = df[["visits", "nvisits", "ovisits", "novisits", "age"]]
dims_2 = df[["visits", "nvisits", "ovisits", "novisits", "income"]]
scatter_matrix_fig = px.scatter_matrix(df, dimensions=dims_1, color="region")
scatter_matrix_fig_2 = px.scatter_matrix(df, dimensions=dims_2, color="region")

bar_fig = px.bar(df, x='age', y='income', title="Bar chart to display Age vs Income Data using color map",
                 hover_data=['region', 'married'], color='gender',
                 labels={'employed': 'employement status'}, height=400)


bar_fig_age_income = px.bar(category_df, x="age category", y="count", color="income category", barmode="stack",
                            title="Age Category vs Income Category", labels={"count": "Total Value"})


bar_fig_region_gender = px.bar(category_df, x="region", y="count", color="gender", barmode="group", title="Region vs Gender",
                               labels={"count": "Total Value"})


histogram_fig_1 = px.histogram(category_df, x="count", marginal="rug", title="Distribution of Values", color='gender',
                               hover_data=category_df.columns, labels={"count": "Value"})


histogram_fig_2 = make_subplots(rows=3, cols=2,
                                subplot_titles=["Region Distribution", "Age Category Distribution",
                                                "Income Category Distribution",
                                                "Health Distribution",
                                                "Gender Distribution",])

histogram_fig_2.add_trace(
    go.Bar(x=category_df["region"], y=category_df["count"]),
    row=1, col=1
)


histogram_fig_2.add_trace(
    go.Bar(x=category_df["age category"], y=category_df["count"]),
    row=1, col=2
)


histogram_fig_2.add_trace(
    go.Bar(x=category_df["income category"], y=category_df["count"]),
    row=2, col=1
)


histogram_fig_2.add_trace(
    go.Bar(x=category_df["health"], y=category_df["count"]),
    row=2, col=2
)


histogram_fig_2.add_trace(
    go.Bar(x=category_df["gender"], y=category_df["count"]),
    row=3, col=1
)

histogram_fig_2.update_layout(
    height=500,
    showlegend=True,
    title_text="Categorical Distributions (Counts)",
    template="plotly_white"
)

pie_fig = px.pie(category_df, values='count', names='region', title='Category',
                 hover_data=['gender', 'age category', 'income category'])


box_fig_1 = px.box(df, y=["visits", "nvisits", "ovisits", "novisits", "age"], color="region")
box_fig_2 = px.box(df, y=["income"], color="region")

violin_fig_1 = px.violin(df, y=["visits", "nvisits", "ovisits", "novisits", "age"], color="region")
violin_fig_2 = px.violin(df, y=["income"], color="region")

sunburst_fig = px.sunburst(df, path=['region', 'health', 'gender'], title="Sunburst plot to show the data region wise",
                           color='age', hover_data=['income'],
                           color_continuous_scale='RdBu')

treemap_fig = px.treemap(df, path=['region', 'health', 'insurance', 'school', 'gender'], title="Tree map to display holistic view",
                         color='age', hover_data=['income'],
                         color_continuous_scale='Spectral')

parallel_fig = px.parallel_categories(df, dimensions=['region', 'gender', 'health'],
                                      color="age", color_continuous_scale=px.colors.sequential.Inferno,
                                      labels={'region': 'midwest', 'gender': 'male', 'health': 'average'})

heatmap_fig = px.density_heatmap(category_df, x="count", y="income category", facet_row="gender", facet_col="age category")

# Returns a matrix of all numeric columns
correlation_matrix = df.select_dtypes(include="number").corr()
corr_fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="portland", zmin=1, zmax=1, title="Correlation Plot")

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(age_gender_fig, "Age and Gender Distribution"),
    plotRenderer.plot_to_card(age_income_fig, "Age Category vs Income Category Distribution"),
    plotRenderer.plot_to_card(income_gender_fig, "Income Distribution by Gender"),
    plotRenderer.plot_to_card(region_income_fig, "Regional Income Distribution"),
    plotRenderer.plot_to_card(region_health_fig, "Region vs Health Distribution"),
    plotRenderer.plot_to_card(health_gender_fig, "Health vs Gender Distribution"),
    plotRenderer.plot_to_card(health_age_fig, "Health vs Age Category Distribution"),
    plotRenderer.plot_to_card(scatter_fig, " Scatter Plot"),
    plotRenderer.plot_to_card(scatter_matrix_fig, " Scatter Matrix (Pair Plot) Plot"),
    plotRenderer.plot_to_card(scatter_matrix_fig_2, " Scatter Matrix Plot 2"),
    plotRenderer.plot_to_card(bar_fig, " Bar Graph"),
    plotRenderer.plot_to_card(bar_fig_age_income, " Stacked Bar Graph"),
    plotRenderer.plot_to_card(bar_fig_region_gender, " Grouped Bar Graph"),
    plotRenderer.plot_to_card(histogram_fig_1, " Histogram Graph"),
    plotRenderer.plot_to_card(histogram_fig_2, " Histogram Sub Graph"),
    plotRenderer.plot_to_card(pie_fig, " Pie Graph"),
    plotRenderer.plot_to_card(box_fig_1, " Box Graph"),
    plotRenderer.plot_to_card(box_fig_2, " Box Graph"),
    plotRenderer.plot_to_card(violin_fig_1, " Violin Graph"),
    plotRenderer.plot_to_card(violin_fig_2, " Violin Graph"),
    plotRenderer.plot_to_card(sunburst_fig, " Sun Burst Plot"),
    plotRenderer.plot_to_card(treemap_fig, " Treemap Plot"),
    plotRenderer.plot_to_card(parallel_fig, " Parallel Plot"),
    plotRenderer.plot_to_card(heatmap_fig, " Density Heatmap Plot"),
    plotRenderer.plot_to_card(corr_fig, " Correlation Heatmap Plot"),
]))

capstone_1_insights = """
- Data is divided into 4 major categories.
- Categories are midwest, northwest, west and other.
- 13.44% memory optimized after assigning correct data type.
- Given data set can be categorized into 3 age categories.Age Category: Senior (>60), Super Senior (>75), Ultra Senior (>100)
- Salary data cane be categoriged into 3, Low (<40KUSD), Medium (<80KUSD) and High (>80KUSD).
- 59.65% data from females and females health condition is poor than males.
- Income of the females are less than male. This could also reason of not having good health.
- Female health awareness program should be conducted.
- Others are midwest are the region where we should conduct more health awareness program.
------------------------------------------------
"""
content.append(builder.card("Inights after Data Visulaization", builder.render_pre(capstone_1_insights)))

html_doc = builder.build_page("Capstone-1 Data Visualization", "\n".join(content))

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "capstone_1_Data_Visualization_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
