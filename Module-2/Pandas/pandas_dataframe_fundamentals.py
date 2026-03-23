import pandas as pd
import numpy as np
from lib.arrays_html import arrays_table_html, arrays_index_report_html
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
    # ...


# Creating a DataFrame from a dictionary
data_dict = {'Name': ['Alice', 'Bob', 'Charlie'],
             'Age': [25, 30, 22],
             'Salary': [50000, 60000, 45000]}

df_dict = pd.DataFrame(data_dict)

# Creating a DataFrame from lists
data_list = [['Alice', 25, 50000], ['Bob', 30, 60000], ['Charlie', 22, 45000]]

# Defining column names
columns_name = ['Name', 'Age', 'Salary']

df_list = pd.DataFrame(data_list, columns=columns_name)

data_array = np.array([['Alice', 25, 50000],
                       ['Bob', 30, 60000],
                       ['Charlie', 22, 45000]])

df_array = pd.DataFrame(data_array, columns=columns_name)

pairs = [
    ("Dataframe created by Dictionary is:", df_dict),
    ("Type of the Dataframe created by Dictionary is:", type(df_dict)),
    ("Dataframe description are:", df_dict.describe().to_dict()),
    ("Shape of the Dataframe is:", df_dict.shape),
    ("Dataframe created by Lists is:", df_list),
    ("Type of the Dataframe created by Lists is:", type(df_list)),
    ("Dataframe description are:", df_list.describe().to_dict()),
    ("Shape of the Dataframe created by Lists is:", df_list.shape),
    ("Dataframe created by Numpy Array is:", df_array),
    ("Type of the Dataframe created by Numpy Array is:", type(df_array)),
    ("Dataframe description are:", df_array.describe().to_dict()),
    ("Shape of the Dataframe created by Numpy Array is:", df_array.shape),
]

# 1) Just the fragment (embed in an existing page or notebook cell)
fragment = arrays_table_html(pairs)

# 2) Full standalone page
html_doc = arrays_index_report_html(
    pairs, page_title="Pandas Dataframe Fundamentals Report")

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
