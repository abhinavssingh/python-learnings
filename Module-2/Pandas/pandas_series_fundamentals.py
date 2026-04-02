import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Pandas Series Fundamentals report...")
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

builder = HtmlBuilder()
html_doc = builder.build_page("Pandas Series Fundamentals Report", builder.grid([
    builder.card("Original series is:", builder.render_series(series)),
    builder.card("Type of the Series is:", builder.render_dict(
        {"dtype": str(series.dtype)})),
    builder.card("Series description are:",
                 builder.render_dict(series.describe().to_dict())),
    builder.card("Shape of the Series is:",
                 builder.render_dict({"shape": series.shape})),
    builder.card("Unique values in the series are:",
                 builder.render_series(series.unique())),
    builder.card("No of unique values in the Series are:",
                 builder.render_series([series.nunique()])),
    builder.card("First 5 value of the Series are:",
                 builder.render_series(series.head(5))),
    builder.card("Last 4 values of the Series are:",
                 builder.render_series(series.tail(4))),
    builder.card("Series with pre defined index is:",
                 builder.render_series(series_with_defined_index)),
    builder.card("Type of the pre defined index is:", builder.render_dict(
        {"type": type(series_with_defined_index).__name__})),
    builder.card("Description of the pre defined index are:", builder.render_dict(
        series_with_defined_index.describe().to_dict())),
    builder.card("Dictionary Series is:", builder.render_series(dict_series)),
    builder.card("Type of the Dictionary Series is:",
                 builder.render_dict({"type": type(dict_series).__name__})),
    builder.card("Description of the Dictionary Series are:",
                 builder.render_dict(dict_series.describe().to_dict())),
    builder.card("Elements from the Dictionary Series based on index position [1:4]:", builder.render_series(
        s_position_values)),
    builder.card("Elements from the Dictionary Series based on index name ['a', 'c', 'e']:", builder.render_series(
        s_index_name_values)),
    builder.card("String Series is:", builder.render_series(string_series)),
    builder.card("Data type of the string series is:",
                 builder.render_dict({"dtype": string_series.dtype})),
    builder.card("Elements from the String Series starts with b:",
                 builder.render_series(string_series_char_pattern)),
    builder.card("Elements from the String Series contains er:",
                 builder.render_series(string_series_char_contains)),
]))


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "pandas_series_fundamentals_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
