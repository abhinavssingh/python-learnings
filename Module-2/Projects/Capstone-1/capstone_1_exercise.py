from lib.data_loader import read_dataset
import io
from lib.html.base import build_html_page
from lib.html.components import full_width_card, grid, card
from lib.html.renderers import render_dataframe_collapsible, render_pre
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running Capstone-1 report...")
    # ...


df = read_dataset("NSMES1988.csv")

content = []

# use for the large dataset
content.append(
    full_width_card(
        "Capstone-1 Data– Interactive Preview",
        render_dataframe_collapsible(df, initial_rows=15)
    )
)

# Get information about the DataFrame
buffer = io.StringIO()
df_info = df.info(buf=buffer)
df_info_str = buffer.getvalue()  # Retrieve the string from the buffer

content.append(
    grid([
        card("Information of the Capstone-1 Dataframe is:", render_pre(df_info_str)),
    ]))

html_doc = build_html_page(
    "Capstone-1 Exercise  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "capstone_1_exercise_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
