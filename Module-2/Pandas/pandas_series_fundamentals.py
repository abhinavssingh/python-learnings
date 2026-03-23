import pandas as pd
import numpy as np
from lib.arrays_html import arrays_table_html, arrays_index_report_html
from lib.report_utils import save_html_report

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

pairs = [
    ("Original series is:", series),
    ("Type of the Series is:", type(series)),
    ("Series description are:", series.describe().to_dict()),
    ("Shape of the Series is:", series.shape),
    ("Unique values in the series are:", series.unique()),
    ("No of unique values in the Series are:", series.nunique()),
    ("First 5 value of the Series are:", series.head(5)),
    ("Last 4 values of the Series are:", series.tail(4)),
    ("Series with pre defined index is:", series_with_defined_index),
    ("Type of the pre defined index is:", type(series_with_defined_index)),
    ("Description of the pre defined index are:", series_with_defined_index.describe().to_dict()),
    ("Dictionary Series is:", dict_series),
    ("Type of the Dictionary Series is:", type(dict_series)),
    ("Description of the Dictionary Series are:", dict_series.describe().to_dict()),
    (" Elemnents from the Dictionary Series based on index position [1:4]", s_position_values),
    (" Elemnents from the Dictionary Series based on index name ['a', 'c', 'e']", s_index_name_values),
    (" String Series is:", string_series),
    (" Data type of the string series is:", string_series.dtype),
    (" Elements from the String Series starts with b", string_series_char_pattern),
    (" Elements from the String Series contains er", string_series_char_contains),
]

# 1) Just the fragment (embed in an existing page or notebook cell)
fragment = arrays_table_html(pairs)

# 2) Full standalone page
html_doc = arrays_index_report_html(pairs, page_title="Pandas Series Operation Report")

# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_series_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")