import numpy as np

from lib.html import HtmlBuilder
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Capstone-1 report...")
    # ...

# Build full column-wise page


builder = HtmlBuilder()

df, report = dl.read_dataset("NSMES1988.csv", optimize=True, handle_unnamed="drop", return_report=True)
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

content.append(
    builder.grid([
        builder.card("Information of the Capstone-1 Unmodified Dataframe is:", builder.render_pre(df_info_str)),
        builder.card("Basic description of modified Dateframe:", builder.render_dict(df_copy.describe().to_dict())),
        builder.card("Optimized Dataframe report:", builder.render_pre(report)),
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
