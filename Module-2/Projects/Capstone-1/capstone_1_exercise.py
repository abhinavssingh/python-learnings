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

df.rename(columns={
    "age": "Age in years (divided by 10)",
    "income": "Family income in USD 10000"
}, inplace=True)

# convert column name to title case
df.columns = [col.title() for col in df.columns]

df_numeric_uint = df.select_dtypes(include=["uint8", "uint16"])
df_numeric_float = df.select_dtypes(include=["float16", "float32"])
df.to_json("NSMES1988.json", orient="records", lines=True)  # Save as JSON for future use
content = []

# use for the large dataset
content.append(
    builder.card(
        "Capstone-1 Data– Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(
    builder.card(
        "Integer Dataframe", builder.render_dataframe_collapsible(df_numeric_uint))
)

content.append(
    builder.card(
        "Float Dataframe", builder.render_dataframe_collapsible(df_numeric_float)))

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
