from lib.html import HtmlBuilder
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running marketing campaign report...")
    # ...


df, report = dl.read_dataset("marketing_data.csv", optimize=True, handle_unnamed="drop", return_report=True)

content = []
builder = HtmlBuilder()

# Get information about the DataFrame
df_info_str = dfh.get_dataframe_info_str(df)

# use for the large dataset
content.append(
    builder.full_width_card(
        "Marketing Campaign Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(
    builder.grid([
        builder.card("Information of the Marketing Dataframe is:", builder.render_pre(df_info_str)),
        builder.card("Optimized Dataframe report:", builder.render_pre(report)),
    ])
)

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
