import numpy as np
import pandas as pd
import plotly.express as px

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running marketing campaign report...")
    # ...


# initialization and set variable
content = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()

# read daraframe
df, report = dl.read_dataset("marketing_data.csv", optimize=True, handle_unnamed="drop", return_report=True)

# Income column has white space to trim the white space
df.columns = df.columns.str.strip()
df_copy = df.copy()

# Ensure Income is numeric (remove $ and commas if needed)
df_copy['Income'] = df_copy['Income'].replace('[\\$,]', '', regex=True).astype(float)

# Group-based mean imputation
df_copy['Income'] = df_copy['Income'].fillna(df_copy.groupby(['Education', 'Marital_Status'])['Income'].transform('mean'))
log_income = df_copy["Income"].transform(lambda x: np.round(np.log10(x), 2) if x > 0 else np.nan)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Income", new_col="Log_Income", values=log_income, inplace=True)

df_copy['Dt_Customer'] = pd.to_datetime(df_copy['Dt_Customer'])

# Create variables to represent the total number of children, age, and total spending.
df_copy = dfh.insert_column_after(
    df_copy, after_col="Teenhome", new_col="Totalchildren", values=df_copy['Kidhome'] + df_copy['Teenhome'], inplace=True)

df_copy = dfh.insert_column_after(
    df_copy, after_col="Year_Birth", new_col="Age", values=df_copy["Dt_Customer"].dt.year - df_copy['Year_Birth'], inplace=True)

# Select columns that contain 'Mnt' and calculate row-wise sum
Total_Mnt = df_copy.loc[:, df_copy.columns.str.contains('Mnt')].sum(axis=1)

df_copy = dfh.insert_column_after(
    df_copy, after_col="MntGoldProds", new_col="TotalSpend", values=Total_Mnt, inplace=True)


age_category = pd.qcut(df_copy['Age'], q=4, labels=['Young', 'Mid-Age', 'Senior', 'Elder'])
df_copy = dfh.insert_column_after(
    df_copy, after_col="Age", new_col="Age_Category", values=age_category, inplace=True)

income_category = pd.qcut(df_copy['Income'], q=4, labels=['Low Income', 'Lower-Middle', 'Upper-Middle', 'High Income'])
df_copy = dfh.insert_column_after(
    df_copy, after_col="Income", new_col="Income_Category", values=income_category, inplace=True)


# there are 3 channels web, catalogue and store

channel_cols = [
    'NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases'
]

Total_Purchases_per_Customer = df_copy[channel_cols].sum(axis=1)
df_copy = dfh.insert_column_after(
    df_copy, after_col="NumStorePurchases", new_col="TotPurchase", values=Total_Purchases_per_Customer, inplace=True)

total_purchases_by_channel = df_copy[channel_cols].sum()

# Get information about the DataFrame
df_info_str = dfh.get_dataframe_info_str(df)
df_copy_info_str = dfh.get_dataframe_info_str(df_copy)

df_outliers = dfh.find_iqr_outliers(df_copy, columns=["Log_Income", "Age"], groupby="Country")
df_clean = dfh.remove_outliers(df_outliers)
total_purchases_by_channel_without_outliers = df_clean[channel_cols].sum()

# data visualization
hist_box_fig_1 = px.histogram(df_copy, x="Log_Income", color="Education", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph, Education as category with Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_2 = px.histogram(df_clean, x="Log_Income", color="Education", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Education as category without Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_3 = px.histogram(df_copy, x="Log_Income", color="Marital_Status", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Marital Status as category with Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_4 = px.histogram(df_clean, x="Log_Income", color="Marital_Status", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Marital Status as category without Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_5 = px.histogram(df_copy, x="Log_Income", color="Age_Category", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Age as category with Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_6 = px.histogram(df_clean, x="Log_Income", color="Age_Category", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Age as category without Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_7 = px.histogram(df_copy, x="Log_Income", color="Country", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Country as category with Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_8 = px.histogram(df_clean, x="Log_Income", color="Country", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Country as category without Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)

corr_columns = ['Age', 'Income', 'Totalchildren', 'TotalSpend']
corr_with_outliers_fig = px.imshow(
    df_copy[corr_columns].corr(),
    text_auto='.2f',
    color_continuous_scale='RdBu',
    title='Correlation Matrix with Outliers'
)

corr_without_ouliers_fig = px.imshow(
    df_clean[corr_columns].corr(),
    text_auto='.2f',
    color_continuous_scale='Oxy',
    title='Correlation Matrix without Outliers'
)

insights_pre = """
1. During the dataframe read only, i have optimized the dataframe
means changing the datatype.
2. There was an empty space in the Income column that i removed.
3. Removed $ and comma from Income column value.
4. Converted Dt_Customer column to datetime.
5. There were 24 places where Income vale was missing. So, I have
updated as per their education/marital_status groups average.
6. Performed Outlier techniques and removed outliears.
7. Visualize the data with and without outliers


"""

# use for the large dataset
content.append(
    builder.full_width_card(
        "Original Marketing Campaign Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(
    builder.full_width_card(
        "Modified Marketing Campaign Interactive Preview",
        builder.render_dataframe_collapsible(df_copy, initial_rows=15)
    )
)

content.append(
    builder.full_width_card(
        "Outliers Dataframe",
        builder.render_dataframe_collapsible(df_outliers, initial_rows=15)
    )
)

content.append(
    builder.full_width_card(
        "Cleaned Dataframe after feature engineering operations",
        builder.render_dataframe_collapsible(df_clean, initial_rows=15)
    )
)


content.append(
    builder.grid(
        [
            builder.card("Information of the Original Marketing Dataframe is:", builder.render_pre(df_info_str)),
            builder.card("Optimized Dataframe report:", builder.render_pre(report)),
            builder.card("Information of the Modified Marketing Dataframe is:", builder.render_pre(df_copy_info_str)),
            builder.card("Description of the Modified Marketing Dataframe is:", builder.render_dict(df_copy.select_dtypes(
                include=["number"]).describe().to_dict())),
            builder.card("Total Spend channel wise with outliers", builder.render_series(total_purchases_by_channel)),
            builder.card("Description of the Cleaned Marketing Dataframe is:", builder.render_dict(df_clean.select_dtypes(
                include=["number"]).describe().to_dict())),
            builder.card("Total Spend channel wise without outliers", builder.render_series(total_purchases_by_channel_without_outliers)),
            builder.card("Data Insights after analysis of the Marketing campaign data", builder.render_pre(insights_pre))

        ])
)

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(hist_box_fig_7, " Histogram Box Graph Country as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_8, " Histogram Box Graph Country as category without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_1, " Histogram Box Graph Education as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_2, " Histogram Box Graph Education as category without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_3, " Histogram Box Graph Marital Status as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_4, " Histogram Box Graph Marital Status as category without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_5, " Histogram Box Graph Age as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_6, " Histogram Box Graph Age as category without Outliers"),
    plotRenderer.plot_to_card(corr_with_outliers_fig, " Correlation Heatmap with Outliers"),
    plotRenderer.plot_to_card(corr_without_ouliers_fig, " Correlation Heatmap without Outliers"),
]))

html_doc = builder.build_page(
    "Marketing Campaign  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "marketing_campaign_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
