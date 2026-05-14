import numpy as np
import pandas as pd
import plotly.express as px

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running ml linear regression report...")
    # ...


# initialization and set variable
content = []
dashboard = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()

df, report = dl.read_dataset("weatherHistory.csv", optimize=False, handle_unnamed="drop", return_report=True)

# convert Formatted Date column to date time
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

# add calendar and fiscal year columns as per country
df = dfh.add_fiscal_calendar(df, "Formatted Date", country="Hungary", calendar_fields={
    "week_of_year", "year"}, fiscal_fields={"fiscal_year", "fiscal_quarter", })

# as optimization is not done so checkinh is contains null
result = dfh.check_nan_inf(df)

df['Precip Type'] = df.groupby(['Year', 'Week_Of_Year'])['Precip Type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
result_after_fillna = dfh.check_nan_inf(df)

# find outliers
num_columns = ["Temperature (C)", "Apparent Temperature (C)", "Humidity", "Wind Speed (km/h)",
               "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]
df_outliers = dfh.find_iqr_outliers(df, columns=num_columns)
df_clean = dfh.remove_outliers(df_outliers)

hist_box_fig_1 = px.histogram(df, x="Temperature (C)", marginal="box", opacity=0.7, barmode="overlay",
                              title="Temperature (C) Histogram Box Graph", hover_data=df.columns)
hist_box_fig_1_clean = px.histogram(df_clean, x="Temperature (C)", marginal="box", opacity=0.7, barmode="overlay",
                                    title="Temperature (C) Histogram Box Graph without outliers", hover_data=df.columns)
hist_box_fig_2 = px.histogram(df, x="Apparent Temperature (C)", marginal="box", opacity=0.7, barmode="overlay",
                              title="Apparent Temperature (C) Histogram Box Graph",)
hist_box_fig_2_clean = px.histogram(df_clean, x="Apparent Temperature (C)", marginal="box", opacity=0.7, barmode="overlay",
                                    title="Apparent Temperature (C) Histogram Box Graph without outliers",)
hist_box_fig_3 = px.histogram(df, x="Humidity", marginal="box", opacity=0.7, barmode="overlay",
                              title="Humidity Histogram Box Graph", hover_data=df.columns)
hist_box_fig_3_clean = px.histogram(df_clean, x="Humidity", marginal="box", opacity=0.7, barmode="overlay",
                                    title="Humidity Histogram Box Graph without outliers", hover_data=df.columns)
hist_box_fig_4 = px.histogram(df, x="Wind Speed (km/h)", marginal="box", opacity=0.7, barmode="overlay",
                              title="Wind Speed (km/h) Histogram Box Graph")
hist_box_fig_4_clean = px.histogram(df_clean, x="Wind Speed (km/h)", marginal="box", opacity=0.7, barmode="overlay",
                                    title="Wind Speed (km/h) Histogram Box Graph without outliers")
hist_box_fig_5 = px.histogram(df, x="Wind Bearing (degrees)", marginal="box", opacity=0.7, barmode="overlay",
                              title="Wind Bearing (degrees) Histogram Box Graph", hover_data=df.columns)
hist_box_fig_5_clean = px.histogram(df_clean, x="Wind Bearing (degrees)", marginal="box", opacity=0.7, barmode="overlay",
                                    title="Wind Bearing (degrees) Histogram Box Graph without outliers", hover_data=df.columns)
hist_box_fig_6 = px.histogram(df, x="Visibility (km)", marginal="box", opacity=0.7, barmode="overlay",
                              title="Visibility (km) Histogram Box Graph",)
hist_box_fig_6_clean = px.histogram(df_clean, x="Visibility (km)", marginal="box", opacity=0.7, barmode="overlay",
                                    title="Visibility (km) Histogram Box Graph without outliers",)
hist_box_fig_7 = px.histogram(df, x="Pressure (millibars)", marginal="box", opacity=0.7, barmode="overlay",
                              title="Pressure (millibars) Histogram Box Graph",)
hist_box_fig_7_clean = px.histogram(df_clean, x="Pressure (millibars)", marginal="box", opacity=0.7, barmode="overlay",
                                    title="Pressure (millibars) Histogram Box Graph without outliers",)
corr_fig_original = px.imshow(df[num_columns].corr(), text_auto='.2f',
                              color_continuous_scale='RdBu', title='Correlation Matrix with Outliers')
corr_fig_clean = px.imshow(df_clean[num_columns].corr(), text_auto='.2f',
                           color_continuous_scale='viridis', title='Correlation Matrix without Outliers')
pair_plot_fig = px.scatter_matrix(df_clean[["Temperature (C)", "Apparent Temperature (C)", "Visibility (km)"]])

# use for the large dataset
content.append(builder.full_width_card("Original Weather History Data",
                                       builder.render_dataframe_collapsible(df, initial_rows=15)))
content.append(
    builder.grid([
        builder.card("Dataframe Null report:", builder.render_dataframe(result)),
        builder.card("Dataframe description report:", builder.render_dict(df.select_dtypes(include="number").describe().to_dict())),
        builder.card("Dataframe Null report after filling null value:", builder.render_dataframe(result_after_fillna)),
        builder.card("Dataframe description report after removing outliers:", builder.render_dict
                     (df_clean.select_dtypes(include="number").describe().to_dict())),
    ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(hist_box_fig_1, " Temperature (C) Histogram Box Graph"),
    plotRenderer.plot_to_card(hist_box_fig_1_clean, " Temperature (C) Histogram Box Graph without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_2, " Apparent Temperature (C) Histogram Box Graph"),
    plotRenderer.plot_to_card(hist_box_fig_2_clean, " Apparent Temperature (C) Histogram Box Graph without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_3, " Humidity Histogram Box Graph"),
    plotRenderer.plot_to_card(hist_box_fig_3_clean, " Humidity Histogram Box Graph without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_4, " Wind Speed(km/h) Histogram Box Graph"),
    plotRenderer.plot_to_card(hist_box_fig_4_clean, " Wind Speed(km/h) Histogram Box Graph without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_5, " Wind Bearing (degrees) Histogram Box Graph"),
    plotRenderer.plot_to_card(hist_box_fig_5_clean, " Wind Bearing (degrees) Histogram Box Graph without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_6, " Visibility (km) Histogram Box Graph"),
    plotRenderer.plot_to_card(hist_box_fig_6_clean, " Visibility (km) Histogram Box Graph without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_7, " Pressure (millibars) Histogram Box Graph"),
    plotRenderer.plot_to_card(hist_box_fig_7_clean, " Pressure (millibars) Histogram Box Graph without Outliers"),
    plotRenderer.plot_to_card(corr_fig_original, " Correlation matrix with Outliers"),
    plotRenderer.plot_to_card(corr_fig_clean, " Correlation matrix without Outliers"),
    plotRenderer.plot_to_card(pair_plot_fig, " Pair Plot"),

]))

html_doc = builder.build_page(
    "ML Linear Regression Report",
    "\n".join(content))

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "ml_linear_regression_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)

print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
