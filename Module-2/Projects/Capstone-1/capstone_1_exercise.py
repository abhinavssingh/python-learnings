from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru
from lib.utility.dataframe.df_helper import DataFrameHelper


def main():
    # your current script code goes here
    print("Running Capstone-1 report...")
    # ...

# Build full column-wise page


builder = HtmlBuilder()

df = dl.read_dataset("NSMES1988.csv")

content = []

# use for the large dataset
content.append(
    builder.card(
        "Capstone-1 Data– Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

# Get information about the DataFrame

df_info_str = DataFrameHelper.get_dataframe_info_str(df)

content.append(
    builder.grid([
        builder.card("Information of the Capstone-1 Dataframe is:", builder.render_pre(df_info_str)),
    ]))

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
