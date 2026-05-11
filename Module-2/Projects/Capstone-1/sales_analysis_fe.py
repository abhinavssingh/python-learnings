import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats.mstats import winsorize

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
dashboard = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()

# read dataframe
df, report = dl.read_dataset("AusApparalSales4thQrt2020.csv", optimize=True, handle_unnamed="drop", return_report=True)
df_copy = df.copy()  # create a copy of the original dataframe for further analysis and visualization


# convert date column to date time
df_copy['Date'] = pd.to_datetime(df_copy['Date'])
# add few columns to analyze the data based on year, month, quarter and fiscal year
df_copy = dfh.add_fiscal_calendar(df_copy, "Date", "Australia",
                                  calendar_fields={"year", "month_name", "day_name", "week_of_month", "weekday", "is_weekend"},
                                  fiscal_fields={"fiscal_year", "fiscal_quarter", "fq_month_range"},
                                  )

unit_bins = [1, 10, 20, 35, 50, 65]
unit_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

unit_category = pd.cut(
    df_copy["Unit"],
    bins=unit_bins,
    labels=unit_labels,
    include_lowest=True
)

sales_bins = [0, 25000, 50000, 80000, 120000, 162500]
sales_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

sales_category = pd.cut(
    df_copy["Sales"],
    bins=sales_bins,
    labels=sales_labels,
    include_lowest=True
)

df_copy = dfh.insert_column_after(
    df_copy, after_col="Unit", new_col="Unit Category", values=unit_category, inplace=True)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Sales", new_col="Sales Category", values=sales_category, inplace=True)

# log transform the sales column to handle skewness and outliers in the data,
# which can help in better visualization and analysis. We will create a new column 'Log_Sales' for this purpose.
log_unit = df_copy["Unit"].transform(lambda x: np.round(np.log10(x), 2) if x > 0 else np.nan)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Unit", new_col="Log_Unit", values=log_unit, inplace=True)
log_sales = df_copy["Sales"].transform(lambda x: np.round(np.log10(x), 2) if x > 0 else np.nan)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Sales", new_col="Log_Sales", values=log_sales, inplace=True)

df_winsorized = df_copy.copy()  # create another copy for winsorization and outlier handling

df_winsorized["Sales_winsor"] = (df_winsorized.groupby("State")["Log_Sales"].transform(lambda x: winsorize(x, limits=(0.05, 0.05))))
df_winsorized["Unit_winsor"] = (df_winsorized.groupby("State")["Log_Unit"].transform(lambda x: winsorize(x, limits=(0.05, 0.05))))

df_outliers = dfh.find_iqr_outliers(df_copy, columns=["Log_Sales", "Log_Unit"], groupby="State")
df_clean = dfh.remove_outliers(df_outliers)

# data visualization

# Create box subplot layout
box_fig_1 = px.box(df_copy, x="State", y=["Log_Unit"], points="all")  # Box plot for Log_Unit with outliers
box_fig_2 = px.box(df_copy, x="State", y=["Log_Sales"], points="all")  # Box plot for Log_Sales with outliers

clean_box_fig_1 = px.box(df_clean, x="State", y=["Log_Unit"], color="Year", points="all")
clean_box_fig_2 = px.box(df_clean, x="State", y=["Log_Sales"], color="Year", points="all")

unit_box_fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Box Unit with Outliers", "Box Units Without Outliers"],
    shared_xaxes=False
)

for trace in box_fig_1.data:
    unit_box_fig.add_trace(trace, row=1, col=1)

for trace in clean_box_fig_1.data:
    unit_box_fig.add_trace(trace, row=1, col=2)

sales_box_fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Box Sales with Outliers", "Box Sales Without Outliers"],
    shared_xaxes=False
)

for trace in box_fig_2.data:
    sales_box_fig.add_trace(trace, row=1, col=1)

for trace in clean_box_fig_2.data:
    sales_box_fig.add_trace(trace, row=1, col=2)

    unit_box_fig.update_layout(
        height=500,
        title="Box Plot Comparison: Raw vs Clean Data",
        boxmode="group",   # important for side-by-side boxes
        showlegend=True
    )

    sales_box_fig.update_layout(
        height=500,
        title="Box Plot Comparison: Raw vs Clean Data",
        boxmode="group",   # important for side-by-side boxes
        showlegend=True
    )

pie_fig = px.pie(df_copy, values='Sales', names='State', title='State Wise Sales Data with Outliers',
                 hover_data=['Group', 'Month_Name', 'Unit'])
clean_pie_fig = px.pie(df_clean, values='Sales', names='State', title='State Wise Sales Data without Outliers (Cleaned)',
                       hover_data=['Group', 'Month_Name', 'Unit'])

sunburst_df = (
    df_clean.groupby(["State", "Month_Name", "Week_Of_Month", "Day_Name"], as_index=False)
    .agg(Total_Sales=("Sales", "sum"))
)

sunburst_fig = px.sunburst(sunburst_df, path=["State", "Month_Name", "Week_Of_Month", "Day_Name"], values="Total_Sales",
                           title="Sales Distribution (%) by State → Month → Week → Day",)

sunburst_fig.update_traces(textinfo="label+percent parent", hovertemplate="<b>%{label}</b><br>" +
                           "Sales: %{value}<br>" + "Parent %: %{percentParent:.1%}<br>" + "Total %: %{percentRoot:.1%}"
                           )

sunburst_monthly_df = (
    df_clean.groupby(["Month_Name", "Group"], as_index=False)
    .agg(Total_Sales=("Sales", "sum"))
)

monthly_sales_sunburst = px.sunburst(sunburst_monthly_df, path=["Month_Name", "Group"], values="Total_Sales",
                                     title="Monthly Sales Distribution by Customer Group")

monthly_sales_sunburst.update_traces(textinfo="label+percent parent", hovertemplate="<b>%{label}</b><br>" +
                                     "Sales: %{value}<br>" + "Parent %: %{percentParent:.1%}<br>" + "Total %: %{percentRoot:.1%}"
                                     )

sales_histogram_fig_monthly = px.histogram(df_copy, x="Month_Name", color="Sales Category", facet_col="State",
                                           barmode="group", labels={"Month_Name": "Month"}, title="Monthly Sales Analysis using Histogram",)


sales_histogram_clean_fig_monthly = px.histogram(df_clean, x="Month_Name", color="Sales Category", facet_col="State",
                                                 barmode="group", labels={"Month_Name": "Month"}, title="Monthly Sales Analysis using Histogram",)

fq_histogram_fig_fq = px.histogram(df_copy, x="Fiscal_Quarter", color="Sales Category", facet_col="State",
                                   barmode="group", labels={"Fiscal_Quarter": "FQ"}, title="Fiscal Quarterly Sales Analysis using Histogram",)

fq_histogram_clean_fig_fq = px.histogram(df_clean, x="FQ_Month_Range", color="Sales Category", facet_col="State",
                                         barmode="group", labels={"FQ_Month_Range": "FQ-2"}, title="Fiscal Quarterly Sales Analysis using Histogram",)

day_histogram_clean_fig_fq = px.histogram(df_clean, x="Day_Name", color="Group", facet_col="State",
                                          barmode="group", labels={"Day_Name": "Day"}, title="Daily Sales Analysis using Histogram",)

time_histogram_clean_fig_fq = px.histogram(df_clean, x="Time", color="Group", facet_col="State",
                                           barmode="group", title="Time-based Sales Analysis using Histogram",)

state_group_df = (df_clean.groupby(["Group", "State"], as_index=False)["Sales"].sum())
state_group_matrix = state_group_df.pivot(index="State", columns="Group", values="Sales").fillna(0)
state_group_pct = state_group_matrix.div(state_group_matrix.sum(axis=1), axis=0,) * 100

state_group_fig = px.imshow(state_group_pct.round(1), text_auto=".1f", color_continuous_scale="Blues",
                            labels={"x": "Customer Group", "y": "State", "color": "Sales %"
                                    }, title="Sales Distribution (%) by State & Customer Group")

state_group_bar = px.bar(state_group_df, x="State", y="Sales", color="Group",
                         barmode="stack", title="State-wise Sales by Customer Group", labels={"Sales": "Sales (AUD)"})

state_month_df = (df_clean.groupby(["Month_Name", "State"], as_index=False)["Sales"].sum())
state_month_matrix = state_month_df.pivot(index="State", columns="Month_Name", values="Sales").fillna(0)
state_month_matrix_pct = state_month_matrix.div(state_month_matrix.sum(axis=1), axis=0,) * 100

state_month_fig = px.imshow(state_month_matrix_pct, text_auto=True, color_continuous_scale="Oxy",
                            labels={"x": "Month", "y": "State", "color": "Sales %"
                                    }, title="Sales Distribution (%) by State & Month")

time_group_df = (df_clean.groupby(["Group", "Time"], as_index=False)["Sales"].sum())
time_group_matrix = (time_group_df.pivot(index="Time", columns="Group", values="Sales").fillna(0))
time_group_matrix_pct = time_group_matrix.div(time_group_matrix.sum(axis=1), axis=0,) * 100

time_group_fig = px.imshow(time_group_matrix_pct, text_auto=True, color_continuous_scale="Viridis",
                           labels={"x": "Customer Group", "y": "Time of Day", "color": "Sales %"
                                   }, title="Sales Heatmap: Time of Day vs Customer Group")

month_time_df = (df_clean.groupby(["Month_Name", "Time"], as_index=False)["Sales"].sum())
month_time_matrix = month_time_df.pivot(index="Time", columns="Month_Name", values="Sales").fillna(0)
month_time_matrix_pct = month_time_matrix.div(month_time_matrix.sum(axis=1), axis=0,) * 100

month_time_fig = px.imshow(month_time_matrix_pct, text_auto=True, color_continuous_scale="YlOrRd",
                           labels={"x": "Month", "y": "Time of Day", "color": "Sales %"
                                   }, title="Sales Heatmap: Month vs Time of Day")

weekday_state_df = (df_clean.groupby(["Day_Name", "State"], as_index=False)["Sales"].sum())
weekday_state_matrix = weekday_state_df.pivot(index="State", columns="Day_Name", values="Sales").fillna(0)
weekday_state_matrix_pct = weekday_state_matrix.div(weekday_state_matrix.sum(axis=1), axis=0,) * 100

weekday_units_fig = px.imshow(weekday_state_matrix_pct, text_auto=True, color_continuous_scale="Plasma",
                              labels={"x": "Day Name", "y": "State", "color": "Sales %"
                                      }, title="Sales Heatmap: Weekday vs State")

weekly_state_df = (df_clean.groupby(["Week_Of_Month", "State"], as_index=False)["Sales"].sum())
weekly_state_matrix = weekly_state_df.pivot(index="State", columns="Week_Of_Month", values="Sales").fillna(0)
weekly_state_matrix_pct = weekly_state_matrix.div(weekly_state_matrix.sum(axis=1), axis=0,) * 100

weekly_state_fig = px.imshow(weekly_state_matrix_pct, text_auto=True, color_continuous_scale="Geyser",
                             labels={"x": "Week of Month", "y": "State", "color": "Sales %"
                                     }, title="Sales Heatmap: Week of Month vs State")

visual_recommednation = """
    Plotly was chosen over Seaborn because:
    - Interactive dashboards improve decision-making
    - Executives can drill down by State, Group, Time
    - Suitable for large datasets (7478 rows)
    """

kpi_df = pd.DataFrame([
    {
        "KPI": "Total Sales (AUD)",
        "Value": f"{df_clean['Sales'].sum():,.0f}",
        "Business Meaning": "Total revenue generated during Q4 2020"
    },
    {
        "KPI": "Total Units Sold",
        "Value": f"{df_clean['Unit'].sum():,.0f}",
        "Business Meaning": "Total number of units sold across all transactions"
    },
    {
        "KPI": "Average Sales per Transaction",
        "Value": f"{df_clean['Sales'].mean():,.0f}",
        "Business Meaning": "Average order value per transaction"
    },
    {
        "KPI": "Median Sales Value",
        "Value": f"{df_clean['Sales'].median():,.0f}",
        "Business Meaning": "Typical transaction size excluding skew from outliers"
    },
    {
        "KPI": "Sales Standard Deviation",
        "Value": f"{df_clean['Sales'].std():,.0f}",
        "Business Meaning": "Sales variability indicating consistency of revenue"
    },
    {
        "KPI": "Outliers Removed (%)",
        "Value": f"{round(100 * (1 - len(df_clean) / len(df_copy)), 2)}%",
        "Business Meaning": "Data cleaned to improve statistical reliability"
    }
])

kpi_table_fig = go.Figure(
    data=[
        go.Table(
            columnwidth=[0.25, 0.15, 0.60],   # control column widths
            header=dict(values=["<b>KPI</b>", "<b>Value</b>", "<b>Business Meaning</b>"],
                        fill_color="#1f3d4f", font=dict(color="white", size=13), align="left", height=38
                        ),
            cells=dict(
                values=[kpi_df["KPI"], kpi_df["Value"], kpi_df["Business Meaning"]],
                fill_color=[["#f7f9fb", "#ffffff"] * len(kpi_df)],
                font=dict(color="#1f2933", size=12), align="left", height=34
            ))
    ])

kpi_table_fig.update_layout(title="Key Performance Indicators – Sales Summary",
                            margin=dict(l=10, r=10, t=30, b=10))

group_sales = (df_clean.groupby("Group", as_index=False).agg(Total_Sales=("Sales", "sum"))
               .sort_values("Total_Sales", ascending=False))

group_sales_fig = px.bar(group_sales, x="Group", y="Total_Sales", text_auto=".2s", title="Total Sales by Customer Group",
                         labels={"Total_Sales": "Sales (AUD)", "Group": "Customer Group"},)

group_sales_fig.update_layout(yaxis_tickformat=",", height=400)

time_sales = (df_clean.groupby("Time", as_index=False).agg(Total_Sales=("Sales", "sum")).sort_values("Total_Sales", ascending=False))

time_sales_fig = px.bar(time_sales, x="Time", y="Total_Sales", text_auto=".2s", title="Sales by Time of Day",
                        labels={"Time": "Time of Day", "Total_Sales": "Sales (AUD)"},)

time_sales_fig.update_layout(yaxis_tickformat=",", height=450)

# use for the large dataset
content.append(builder.full_width_card("Original Sales Analysis for AAL Interactive Preview",
                                       builder.render_dataframe_collapsible(df, initial_rows=15)))

outlier_rows = df_outliers[
    df_outliers.filter(like="_outlier").any(axis=1)
]

content.append(builder.full_width_card("Modified Dataframe with Log Transformations and Outliers:",
                                       builder.render_dataframe_collapsible(df_outliers, initial_rows=15)))

content.append(builder.full_width_card("Winsorized Dataframe:",
                                       builder.render_dataframe_collapsible(df_winsorized, initial_rows=15)))
content.append(builder.full_width_card("Cleaned Dataframe after removing Outliers and Adding Fiscal Fields:",
                                       builder.render_dataframe_collapsible(df_clean, initial_rows=15)))

content.append(
    builder.grid([
        builder.card("Optimized Dataframe report:", builder.render_pre(report)),
        builder.card("Dataframe description report:", builder.render_dict(df.select_dtypes(include="number").describe().to_dict())),
        builder.card("Unique Value for Unit:", builder.render_series(sorted(df["Unit"].unique()))),
        builder.card("Unique Value for Sales:", builder.render_series(sorted(df["Sales"].unique()))),
        builder.card("Removed Outliers:", builder.render_dataframe(outlier_rows)),
        builder.card("Cleaned Dataframe description report:", builder.render_dict(df_clean.select_dtypes(include="number").describe().to_dict())),
        builder.card("Sunburst data:", builder.render_dataframe(sunburst_df)),
        builder.card("State vs Customer Group data:", builder.render_dataframe(state_group_df)),
        builder.card("State vs Month data:", builder.render_dataframe(state_month_df)),
        builder.card("Time vs Customer Group data:", builder.render_dataframe(time_group_df)),
        builder.card("Month vs Time data:", builder.render_dataframe(month_time_df)),
        builder.card("Weekday vs State data:", builder.render_dataframe(weekday_state_df)),
        builder.card("Week of Month vs State data:", builder.render_dataframe(weekly_state_df)),
        builder.card("Dashboard Navigation",
                     """<a href="ceo_head_of_sales_dashboard.html" target="_blank" style="display:inline-block;padding:12px 18px;background:#1f3b4d;color:white;
               text-decoration:none;border-radius:6px;font-weight:600;">📈 View Interactive Sales Executive Dashboard
        </a>""")
    ]))

dashboard.append(
    builder.full_width_card("Key Performance Indicators – Sales Summary", builder.render_dataframe_collapsible(kpi_df, 5))
)

dashboard.append(
    builder.grid([
        builder.card("Visualization Recommendation", builder.render_pre(visual_recommednation)),
        builder.card("Dashboard Navigation",
                     """<a href="sales_analysis_report_fe.html" target="_blank" style="display:inline-block;padding:12px 18px;background:#1f3b4d;color:white;
               text-decoration:none;border-radius:6px;font-weight:600;">📈 View Interactive Sales Dashboard
        </a>""")
    ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(unit_box_fig, " Units Box Graph comparison with and without outliers"),
    plotRenderer.plot_to_card(sales_box_fig, " Sales Box Graph comparison with and without outliers"),
    plotRenderer.plot_to_card(pie_fig, " State Wise Sales Data with Outliers"),
    plotRenderer.plot_to_card(clean_pie_fig, " State Wise Sales Data without Outliers (Cleaned)"),
    plotRenderer.plot_to_card(sunburst_fig, " Sales Distribution (%) by State → Month → Week → Day"),
    plotRenderer.plot_to_card(monthly_sales_sunburst, " Monthly Sales Distribution"),
    plotRenderer.plot_to_card(sales_histogram_clean_fig_monthly, " Monthly Sales Analysis using Histogram (Cleaned)"),
    plotRenderer.plot_to_card(fq_histogram_clean_fig_fq, " Fiscal Quarterly Sales Analysis using Histogram (Cleaned)"),
    plotRenderer.plot_to_card(day_histogram_clean_fig_fq, " Daily Sales Analysis using Histogram"),
    plotRenderer.plot_to_card(time_histogram_clean_fig_fq, " Time-based Sales Analysis using Histogram"),
]))

dashboard.append(
    builder.chart_grid([
        plotRenderer.plot_to_card(group_sales_fig, " Total Sales by Customer Group"),
        plotRenderer.plot_to_card(time_sales_fig, " Sales by Time of Day"),
        plotRenderer.plot_to_card(state_group_fig, " Sales Distribution (%) by State & Customer Group"),
        plotRenderer.plot_to_card(state_group_bar, " Sales Distribution (%) by State & Customer Group"),
        plotRenderer.plot_to_card(state_month_fig, " Sales Distribution (%) by State & Month"),
        plotRenderer.plot_to_card(time_group_fig, " Sales Heatmap: Time of Day vs Customer Group"),
        plotRenderer.plot_to_card(month_time_fig, " Sales Heatmap: Month vs Time of Day"),
        plotRenderer.plot_to_card(weekday_units_fig, " Sales Heatmap: Weekday vs State"),
        plotRenderer.plot_to_card(weekly_state_fig, " Sales Heatmap: Week of Month vs State"),
        plotRenderer.plot_to_card(kpi_table_fig, "Key Performance Indicators – Sales Summary"),
    ]))

html_doc = builder.build_page(
    "Sales Analysis  Report",
    "\n".join(content))

dashboard_doc = builder.build_page(
    "CEO Head of Sales Dashboard",
    "\n".join(dashboard))

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "sales_analysis_report_fe.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)

dashboard_path = ru.save_html_report(
    __file__,
    "ceo_head_of_sales_dashboard.html",   # file name
    dashboard_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=False
)
print(f"Wrote report to: {output_path}")
print(f"Wrote report to: {dashboard_path}")

if __name__ == "__main__":
    main()
