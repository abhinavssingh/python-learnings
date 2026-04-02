import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Pandas DataFrame Fundamentals report...")
    # ...

# Build full column-wise page


builder = HtmlBuilder()

# create a date range dataframe
# frequency can be 'D' for daily, 'M' for monthly, etc.
date_range = pd.date_range(start="2024-01-01", end="2026-12-31", freq="D")
df = pd.DataFrame({"date": date_range})

# created a utility function in lib/date_utility.py to add fiscal calendar fields based on country-specific rules
df_india = dfh.add_fiscal_calendar(df, "date", "India")

# to change the data type of the integer columns to int8
df_modified = df_india.copy()
int_cols = df_india.select_dtypes(include=["int"])
df_modified[int_cols.columns] = df_modified[int_cols.columns].astype("int8")

fy_summary = (
    df_india.groupby(["Fiscal_Year", "Fiscal_Quarter"])
            .size()
            .reset_index(name="Days_Count")
)


weekend_summary = (
    df_india.groupby(["Fiscal_Year", "IsWeekend"])
            .size()
            .unstack(fill_value=0)
)

weekend_summary_calendar = (
    df_india.groupby(["Calendar_Quarter", "Year", "IsWeekend"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
)

fy_end_dates = df_india.groupby("Fiscal_Year")["date"].max().reset_index()

# Get information about the DataFrame
df_info_str = dfh.get_dataframe_info_str(df_india)
df_modified_info_str = dfh.get_dataframe_info_str(df_modified)

content = []

content.append(
    builder.full_width_card(
        "Date Range DataFrame",
        builder.render_dataframe_collapsible(df_india, initial_rows=15)
    )
)

# Section — small summaries
content.append(
    builder.grid([
        builder.card("Information of the Date Range Dataframe is:",
                     builder.render_pre(df_info_str)),
        builder.card("Fiscal Year and Quarter Summary",
                     builder.render_dataframe(fy_summary)),
        builder.card("Weekend vs Weekday Count by Fiscal Year",
                     builder.render_dataframe(weekend_summary)),
        builder.card("Weekend vs Weekday Count by Calendar Quarter",
                     builder.render_dataframe(weekend_summary_calendar)),
        builder.card("Data types after modification",
                     builder.render_pre(df_modified_info_str)),
        builder.card("Fiscal Year End Dates",
                     builder.render_dataframe(fy_end_dates))
    ])
)

html_doc = builder.build_page(
    "Pandas Date Range DataFrame Report",
    "\n".join(content)
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "pandas_date_range_dataframe_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
