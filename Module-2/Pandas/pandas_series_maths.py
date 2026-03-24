import numpy as np
import pandas as pd

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_series, render_dict
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
    # ...


# mathematical operation on series
d1 = [4, 5, 6, 7, 8, 9, 10, np.nan]
d2 = [11, 12, 13, 14, 15, 16, 17]
s1 = pd.Series(d1)
s2 = pd.Series(d2)
sum = s1 + s2
sub = s2 - s1
division = (s2 / s1).round(3)
mulitiplication = s1 * s2

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

html_doc = build_html_page("Pandas Series Maths Operation Report", grid([
    card("First series S1 for math operation is:", render_series(s1)),
    card("Second series S2 for math operation is:", render_series(s2)),
    card("Data Type of the Series S1:", render_dict({"dtype": s1.dtype})),
    card("Sum of the two series is:", render_series(sum)),
    card("Subtract of the two series is:", render_series(sub)),
    card("Multiplication of the two series is:", render_series(mulitiplication)),
    card("Division of the two series is:", render_series(division)),
    card("Square of the S1 series is:", render_series(squared_series)),
    card("Default Sort order in series S1 is ascending:", render_series(s1_asc)),
    card("Reverse Sort order S1 series is:", render_series(s1_dsc)),
    card("Has S1 any null:", render_series(is_s1_null)),
    card("After filling S1 series if null exists:", render_series(filled_series)),
    card("Data Type of the Series S1 after filling value:", render_dict({"dtype": filled_series.dtype})),
    card("Elements from S1 series > 6:", render_series(s1_greater_6)),
    card("Elements from S1 series == 8:", render_series(s1_equal_8)),
    card("Elements from S1 series != 6:", render_series(s1_not_equal_6)),
    card("Elements from S1 series values equal to 6, 7, 8:", render_series(s1_range)),
]))


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_series_maths_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
