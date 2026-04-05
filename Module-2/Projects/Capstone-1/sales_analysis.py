import pandas as pd
import plotly.express as px

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running sales analysis report...")
    # ...


# initialization and set variable
content = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()

# read dataframe
df = dl.read_dataset("AusApparalSales4thQrt2020.csv", handle_unnamed="drop")

# convert date column to date time
df['Date'] = pd.to_datetime(df['Date'])

# add few columns to analyze the data based on year, month, quarter and fiscal year
df = dfh.add_fiscal_calendar(df, "Date", "Australia", calendar_fields={
    "year", "month_name", "day_name", "weekday", "is_weekend"}, fiscal_fields={"fiscal_year", "fiscal_quarter"},)

# again optimize the column data type
df = dfh.optimize_numeric_dtypes(df)

# Get information about the DataFrame
df_info_str = dfh.get_dataframe_info_str(df)
correlation_matrix = df[["Unit", "Sales"]].corr()

time_group_df = (df.groupby(["Time", "Group"], as_index=False)["Sales"].sum())
time_group_matrix = (time_group_df.pivot(index="Time", columns="Group", values="Sales").fillna(0))

state_group_df = (df.groupby(["State", "Group"], as_index=False)["Sales"].sum())
state_group_matrix = state_group_df.pivot(index="Group", columns="State", values="Sales").fillna(0)
state_group_pct = state_group_matrix.div(state_group_matrix.sum(axis=1), axis=0,) * 100

state_month_df = (df.groupby(["State", "Month_Name"], as_index=False)["Sales"].sum())
state_month_matrix = state_month_df.pivot(index="Month_Name", columns="State", values="Sales").fillna(0)

month_time_df = (df.groupby(["Month_Name", "Time"], as_index=False)["Sales"].sum())
month_time_matrix = month_time_df.pivot(index="Month_Name", columns="Time", values="Sales").fillna(0)

weekday_state_df = (df.groupby(["Day_Name", "State"], as_index=False)["Sales"].sum())
weekday_state_matrix = weekday_state_df.pivot(index="State", columns="Day_Name", values="Sales").fillna(0)

# data visualization
violin_fig_1 = px.violin(df, x="State", y=["Unit"], color="Year", points="all")
violin_fig_2 = px.violin(df, x="State", y=["Sales"], color="Year", points="all")
histogram_fig_monthly = px.histogram(df, x="Month_Name", color="Sales", facet_col="State",
                                     barmode="group", labels={"Month_Name": "Month"}, title="Monthly Sales Analysis using Histogram",)
histogram_fig_fq = px.histogram(df, x="Fiscal_Quarter", color="Sales", facet_col="State",
                                barmode="group", labels={"Fiscal_Quarter": "FQ"}, title="Fiscal Quarterly Sales Analysis using Histogram",)
pie_fig = px.pie(df, values='Sales', names='State', title='State Wise Sales Data',
                 hover_data=['Group', 'Month_Name', 'Unit'])
sunburst_fig = px.sunburst(df, path=['State', 'Month_Name', 'Group'], title="Sunburst plot to show the data State wise",
                           hover_data='Unit',
                           color_continuous_scale='RdBu')
treemap_fig = px.treemap(df, path=['State', 'Month_Name', 'Group'], title="Tree map to display holistic view of Sales Data",
                         hover_data='Unit',
                         color_continuous_scale='Spectral')
corr_fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="portland", zmin=1, zmax=1, title="Correlation Plot")

state_group_fig = px.imshow(state_group_pct.round(1), text_auto=".1f", color_continuous_scale="Blues",
                            labels={"x": "Customer Group", "y": "State", "color": "Sales %"
                                    }, title="Sales Distribution (%) by State & Customer Group")


state_month_fig = px.imshow(state_month_matrix, text_auto=True, color_continuous_scale="Oxy",
                            labels={"x": "Customer Group", "y": "Month", "color": "Sales"
                                    }, title="Sales Distribution by State & Month")

time_group_fig = px.imshow(time_group_matrix, text_auto=True, color_continuous_scale="Viridis",
                           labels={"x": "Customer Group", "y": "Time of Day", "color": "Total Sales"
                                   }, title="Sales Heatmap: Time of Day vs Customer Group")

month_time_fig = px.imshow(month_time_matrix, text_auto=True, color_continuous_scale="YlOrRd",
                           labels={"x": "Time of Day", "y": "Month", "color": "Total Sales"
                                   }, title="Sales Heatmap: Month vs Time of Day")

weekday_units_fig = px.imshow(weekday_state_matrix, text_auto=True, color_continuous_scale="Plasma",
                              labels={"x": "Sate", "y": "Day Name", "color": "Total Sales"
                                      }, title="Sales Heatmap: Weekday vs State")


# use for the large dataset
content.append(
    builder.full_width_card(
        "Sales Analysis for AAL Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(
    builder.grid([
        builder.card("Information of the Sales Dataframe is:",
                     builder.render_pre(df_info_str)),
        builder.card("Description of the Sales Dataframe is:",
                     builder.render_dict(df.select_dtypes(include="number").describe().to_dict()))
    ])
)

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(violin_fig_1, " Units Violin Graph"),
    plotRenderer.plot_to_card(violin_fig_2, " Sales Violin Graph"),
    plotRenderer.plot_to_card(histogram_fig_monthly, " Monthly Sales Analysis using Histogram"),
    plotRenderer.plot_to_card(histogram_fig_fq, " Fiscal Quarterly Analysis using Histogram"),
    plotRenderer.plot_to_card(pie_fig, " State Wise Sales Data"),
    plotRenderer.plot_to_card(sunburst_fig, " Sunburst plot to show the data State wise"),
    plotRenderer.plot_to_card(treemap_fig, " Tree map to display holistic view of Sales Data"),
    plotRenderer.plot_to_card(time_group_fig, " Heatmap Plot"),
    plotRenderer.plot_to_card(state_group_fig, " Heatmap Plot"),
    plotRenderer.plot_to_card(state_month_fig, " Heatmap Plot"),
    plotRenderer.plot_to_card(weekday_units_fig, " Heatmap Plot"),
    plotRenderer.plot_to_card(month_time_fig, " Heatmap Plot"),
    plotRenderer.plot_to_card(corr_fig, " Correlation Heatmap Plot"),
]))

html_doc = builder.build_page(
    "Sales Analysis  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "sales_analysis_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
