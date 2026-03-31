import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Pandas indexing and slicing report...")
    # ...

# Build full column-wise page


builder = HtmlBuilder()

# 1. Generate a 5x5 array of random integer data
# The values will be random integers between 0 and 19
data = np.random.randint(0, 20, size=(5, 5))

# 2. Define the column labels
column_labels = ['Col_A', 'Col_B', 'Col_C', 'Col_D', 'Col_E']

# 3. Define the row labels (index)
row_labels = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5']

# 4. Create the DataFrame, specifying data, columns, and index labels
df = pd.DataFrame(data, columns=column_labels, index=row_labels)

# 5. Select a single column using its name and a single row using its label
df_col_a = df['Col_A']
df_row_1 = df.loc['Row_1']

# 6. Select multiple columns, rows using a list of column and row names
df_col_a_col_d = df[['Col_A', 'Col_D']]
df_row_1_row_3 = df.loc['Row_1':'Row_3']
df_loc_row_1_col_a = df.loc['Row_1', 'Col_A']
df_iloc_row_4 = df.iloc[3]  # Using iloc for the 4th row (index 3)
df_at_row_2_col_b = df.at['Row_2', 'Col_B']  # Using at for a single value
df_iat_row_3_col_c = df.iat[2, 2]  # Using iat for a single value (Row 3, Col C)

# Condition basis to get the rows and columns based on a condition
# For example, select rows where 'Col_A' is greater than 10
df_col_a_gt_10 = df[df['Col_A'] > 10]
df_row_3_gt_13 = df.iloc[2][df.iloc[2] > 13]
df_col_e_row_2_gt_11 = df[(df['Col_E'] > 11) & (df.index == 'Row_2')]

html_doc = builder.build_page("Pandas Dataframe Indexing and Slicing Report", builder.grid([
    builder.card("Dataframe is:", builder.render_dataframe(df)),
    builder.card("Column A:", builder.render_series(df_col_a)),
    builder.card("Row 1:", builder.render_series(df_row_1)),
    builder.card("Value at Row 1, Column A:", builder.render_dict({"Value": df_loc_row_1_col_a})),
    builder.card("Columns A and D:", builder.render_dataframe(df_col_a_col_d)),
    builder.card("Rows 1 to 3 using loc:", builder.render_dataframe(df_row_1_row_3)),
    builder.card("Row 4 (using iloc):", builder.render_series(df_iloc_row_4)),
    builder.card("Value at Row 2, Column B (using at):", builder.render_dict({"Value": df_at_row_2_col_b})),
    builder.card("Value at Row 3, Column C (using iat):", builder.render_dict({"Value": df_iat_row_3_col_c})),
    builder.card("Rows where Column A > 10:", builder.render_dataframe(df_col_a_gt_10)),
    builder.card("Values in Row 3 > 13:", builder.render_series(df_row_3_gt_13)),
    builder.card("Values in Column E > 11 and Row is Row 2:", builder.render_dataframe(df_col_e_row_2_gt_11)),
]))


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "pandas_dataframe_indexing_slicing_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
