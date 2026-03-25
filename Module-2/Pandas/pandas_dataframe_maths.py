import numpy as np
import pandas as pd

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_dataframe
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
    # ...


data_1 = np.random.randint(0, 10, size=(5, 5))  # Create a 5x5 array of random integers
column_labels = ['Col_A', 'Col_B', 'Col_C', 'Col_D', 'Col_E']  # Define column labels
row_labels = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5']  # Define row labels
df_1 = pd.DataFrame(data_1, columns=column_labels, index=row_labels)  # Create a DataFrame from the array

data_2 = np.random.randint(0, 10, size=(5, 5))  # Create another 5x5 array of random integers
df_2 = pd.DataFrame(data_2, columns=column_labels, index=row_labels)  # Create another DataFrame from the second array
# Perform element-wise addition of the two DataFrames
df_sum = df_1 + df_2
df_subtract = df_1 - df_2
df_multiply = df_1 * df_2
df_divide = df_1 / df_2.replace(0, np.nan)  # Replace zeros with NaN to avoid division by zero
df_power = df_1 ** 2  # Square each element in the first DataFrame
df_apply_func = df_2.apply(np.sqrt)  # Apply the square root function to each element in the first DataFrame
df_apply_func_lambda = df_2.apply(lambda x: x * 2)  # Apply a lambda function to double each element in the second DataFrame

# join the two dataframes using concat
df_concat = pd.concat([df_1, df_2], axis=0)  # Concatenate along rows (axis=0)
df_concat_columns = pd.concat([df_1, df_2], axis=1)  # Concatenate along columns (axis=1)
df_concat_ignore_index = pd.concat([df_1, df_2], axis=0, ignore_index=True)  # Concatenate and ignore index
df_concat_keys = pd.concat([df_1, df_2], axis=0, keys=['DF1', 'DF2'])  # Concatenate with keys for hierarchical indexing

# join the two dataframes using merge
# For merging, we need a common key. Let's create a common key in both dataframes
df_1['Key'] = ['K1', 'K2', 'K3', 'K4', 'K5']
df_2['Key'] = ['K1', 'K2', 'K3', 'K4', 'K5']
df_merged = pd.merge(df_1, df_2, on='Key', suffixes=('_DF1', '_DF2'))  # Merge on the 'Key' column with suffixes to differentiate columns
df_merged_inner = pd.merge(df_1, df_2, on='Key', how='inner', suffixes=('_DF1', '_DF2'))  # Inner join
df_merged_outer = pd.merge(df_1, df_2, on='Key', how='outer', suffixes=('_DF1', '_DF2'))  # Outer join
df_merged_left = pd.merge(df_1, df_2, on='Key', how='left', suffixes=('_DF1', '_DF2'))  # Left join
df_merged_right = pd.merge(df_1, df_2, on='Key', how='right', suffixes=('_DF1', '_DF2'))  # Right join
df_intersection = pd.merge(df_1, df_2, on='Key', how='inner')  # Intersection of the two DataFrames based on the 'Key' column

df_1_sorted = df_1.sort_values(by='Col_A')  # Sort df_1 by 'Col_A'
df_2_sorted = df_2.sort_values(by='Col_C')  # Sort df_2 by 'Col_C'

html_doc = build_html_page("Pandas Dataframe Fundamentals Report", grid([
    card("Dataframe 1:", render_dataframe(df_1)),
    card("Dataframe 2:", render_dataframe(df_2)),
    card("Sum of Dataframes:", render_dataframe(df_sum)),
    card("Difference of Dataframes:", render_dataframe(df_subtract)),
    card("Element-wise Multiplication of Dataframes:", render_dataframe(df_multiply)),
    card("Element-wise Division of Dataframes:", render_dataframe(df_divide)),
    card("Dataframe 1 raised to the power of 2:", render_dataframe(df_power)),
    card("Square root of Dataframe 2:", render_dataframe(df_apply_func)),
    card("Dataframe 2 with each element doubled:", render_dataframe(df_apply_func_lambda)),
    card("Concatenation of Dataframes along rows:", render_dataframe(df_concat)),
    card("Concatenation of Dataframes along columns:", render_dataframe(df_concat_columns)),
    card("Concatenation of Dataframes with keys:", render_dataframe(df_concat_keys)),
    card("Merged Dataframes (Inner Join):", render_dataframe(df_merged_inner)),
    card("Merged Dataframes (Outer Join):", render_dataframe(df_merged_outer)),
    card("Merged Dataframes (Left Join):", render_dataframe(df_merged_left)),
    card("Merged Dataframes (Right Join):", render_dataframe(df_merged_right)),
    card("Intersection of Dataframes based on 'Key':", render_dataframe(df_intersection)),
    card("Dataframe 1 sorted by 'Col_A':", render_dataframe(df_1_sorted)),
    card("Dataframe 2 sorted by 'Col_C':", render_dataframe(df_2_sorted)),
]))


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_dataframe_fundamentals_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
