from lib.data_loader import read_dataset
import numpy as np
import pandas as pd
import io
from lib.html.base import build_html_page
from lib.html.components import full_width_card, grid, card
from lib.html.renderers import render_dataframe_collapsible, render_pre, render_dict
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
    # ...


df = read_dataset("HousePrices.csv")
content = []

# Get information about the DataFrame
buffer = io.StringIO()
df_info = df.info(buf=buffer)
df_info_str = buffer.getvalue()  # Retrieve the string from the buffer

# use for the large dataset
content.append(
    full_width_card(
        "Housing Dataset – Interactive Preview",
        render_dataframe_collapsible(df, initial_rows=15)
    )
)

# Section — small summaries
content.append(
    grid([
        card("Information of the Housing Dataframe is:", render_pre(df_info_str)),
        card("Shape of the Dataframe is ", render_dict({"Shape": df.shape})),
        card("Dataframe description are:", render_dict(df.describe().to_dict())),
    ])
)


html_doc = build_html_page(
    "Pandas Exercise  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_dataframe_exercise_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
