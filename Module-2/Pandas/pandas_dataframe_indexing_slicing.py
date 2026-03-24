import numpy as np
import pandas as pd

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_dataframe, render_dict, render_kv, render_series
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
    # ...


# 1. Generate a 5x5 array of random integer data
# The values will be random integers between 0 and 19
data = np.random.randint(0, 20, size=(5, 5))

# 2. Define the column labels
column_labels = ['Col_A', 'Col_B', 'Col_C', 'Col_D', 'Col_E']

# 3. Define the row labels (index)
row_labels = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5']

# 4. Create the DataFrame, specifying data, columns, and index labels
df = pd.DataFrame(data, columns=column_labels, index=row_labels)

df_col_a = df['Col_A']
df_row_1 = df.loc['Row_1']
df_loc_row_1_col_a = df.loc['Row_1', 'Col_A']
print(df_loc_row_1_col_a)

html_doc = build_html_page("Pandas Dataframe Indexing and Slicing Report", grid([
    card("Dataframe is:", render_dataframe(df)),
    card("Column A:", render_series(df_col_a)),
    card("Row 1:", render_series(df_row_1)),
    card("Value at Row 1, Column A:", render_dict({"Value": df_loc_row_1_col_a})),
]))


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_dataframe_indexing_slicing_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
