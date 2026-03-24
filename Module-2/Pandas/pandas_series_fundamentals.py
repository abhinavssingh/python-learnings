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


data = np.array([1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 8, 1])
series = pd.Series(data)

# Creating a Pandas Series with a specified index
index = ['a', 'b', 'c', 'd', 'e']
data_1 = [1, 2, 3, 4, 5]
series_with_defined_index = pd.Series(data_1, index=index)

# Creating a Pandas Series from a dictionary
data_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
dict_series = pd.Series(data_dict)

# select elements from series based on index position and name
s_position_values = dict_series.iloc[1:4]
s_index_name_values = dict_series.loc[['a', 'c', 'e']]

# Select elements using string methods (if applicable)
string_series = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry'])
string_series_char_pattern = string_series[string_series.str.startswith('b')]
string_series_char_contains = string_series[string_series.str.contains("er")]

html_doc = build_html_page("Pandas Series Fundamentals Report", grid([
    card("Original series is:", render_series(series)),
    card("Type of the Series is:", render_dict({"dtype": str(series.dtype)})),
    card("Series description are:", render_dict(series.describe().to_dict())),
    card("Shape of the Series is:", render_dict({"shape": series.shape})),
    card("Unique values in the series are:", render_series(series.unique())),
    card("No of unique values in the Series are:", render_series([series.nunique()])),
    card("First 5 value of the Series are:", render_series(series.head(5))),
    card("Last 4 values of the Series are:", render_series(series.tail(4))),
    card("Series with pre defined index is:", render_series(series_with_defined_index)),
    card("Type of the pre defined index is:", render_dict({"type": type(series_with_defined_index).__name__})),
    card("Description of the pre defined index are:", render_dict(series_with_defined_index.describe().to_dict())),
    card("Dictionary Series is:", render_series(dict_series)),
    card("Type of the Dictionary Series is:", render_dict({"type": type(dict_series).__name__})),
    card("Description of the Dictionary Series are:", render_dict(dict_series.describe().to_dict())),
    card("Elements from the Dictionary Series based on index position [1:4]:", render_series(s_position_values)),
    card("Elements from the Dictionary Series based on index name ['a', 'c', 'e']:", render_series(s_index_name_values)),
    card("String Series is:", render_series(string_series)),
    card("Data type of the string series is:", render_dict({"dtype": string_series.dtype})),
    card("Elements from the String Series starts with b:", render_series(string_series_char_pattern)),
    card("Elements from the String Series contains er:", render_series(string_series_char_contains)),
]))


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_series_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
