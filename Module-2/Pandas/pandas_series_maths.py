import pandas as pd
import numpy as np
from lib.arrays_html import arrays_table_html, arrays_index_report_html
from lib.report_utils import save_html_report

# mathematical operation on series
d1 = [4, 5, 6, 7, 8, 9, 10, np.nan]
d2 = [11, 12, 13, 14, 15, 16, 17]
s1 = pd.Series(d1)
s2 = pd.Series(d2)
sum = s1 + s2
sub = s2 - s1
division = (s2/s1).round(3)
mulitiplication = s1*s2

# Apply a function to each element
squared_series = s1.apply(lambda x: x**2)

# Sort the Series by values
s1_asc = s1.sort_values()
s1_dsc = s1.sort_values(ascending=False)

# Check for missing values
is_s1_null = s1.isnull()

# Fill missing values with a specified value
filled_series = s1.fillna(0)

# Select elements greater than 6
s1_greater_6 = s1[s1 > 6]

# Select elements equal to 8
s1_equal_8 = s1[s1 == 8]

# Select elements not equal to 6
s1_not_equal_6 = s1[s1 != 6]

# Select elements based on multiple conditions
s1_values = s1[(s1 > 5) & (s1 < 10)]

# Select elements based on a list of values
s1_range = s1[s1.isin([6, 7, 8])]

pairs = [
    (" First series S1 for math operation is:", s1),
    (" Second series S2 for math operation is:", s2),
    (" Data Type of the Series S1:", s1.dtype),
    (" Sum of the two series is:", sum),
    (" Subtract of the two series is:", sub),
    (" Multplication of the two series is:", mulitiplication),
    (" Division of the two series is:", division),
    (" Square of the S1 series is:", squared_series),
    (" Default Sort order in series S1 is ascending:", s1_asc),
    (" Reverse Sort order S1 series is:", s1_dsc),
    (" Has S1 any null:", is_s1_null),
    (" After filling S1 series if null exists:", filled_series),
    (" Data Type of the Series S1 after filling value", filled_series.dtype),
    (" Elemets from S1 series > 6", s1_greater_6),
    (" Elemets from S1 series ==8", s1_equal_8),
    (" Elemets from S1 series !=6", s1_not_equal_6),
    (" Elemets from S1 series values equal to 6, 7, 8", s1_range),
]

# 1) Just the fragment (embed in an existing page or notebook cell)
fragment = arrays_table_html(pairs)

# 2) Full standalone page
html_doc = arrays_index_report_html(pairs, page_title="Pandas Series Maths Operation Report")

# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_series_maths_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")