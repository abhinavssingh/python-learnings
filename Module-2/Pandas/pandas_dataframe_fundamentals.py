import numpy as np
import pandas as pd
import io

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_dataframe, render_dict, render_pre
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
df_array_head = df_array.head(2)  # Get the first few rows of the DataFrame
df_dict_tail = df_dict.tail(1)  # Get the last few rows of the DataFrame
# Get information about the DataFrame
buffer = io.StringIO()
df_dict_info = df_dict.info(buf=buffer)
df_dict_info_str = buffer.getvalue()  # Retrieve the string from the buffer

html_doc = build_html_page("Pandas Dataframe Fundamentals Report", grid([
    card("Dataframe created by Dictionary is:", render_dataframe(df_dict)),
    card("Type of the Dataframe created by Dictionary is:", render_dict({"Type": type(df_dict).__name__})),
    card("Dataframe description are:", render_dict(df_dict.describe().to_dict())),
    card("Information about the Dataframe created by Dictionary is:", render_pre(df_dict_info_str)),
    card("Last row of the Dataframe created by Dictionary:", render_dataframe(df_dict_tail)),
    card("Shape of the Dataframe is:", render_dict({"Shape": df_dict.shape})),
    card("Dataframe created by Lists is:", render_dataframe(df_list)),
    card("Type of the Dataframe created by Lists is:", render_dict({"Type": type(df_list).__name__})),
    card("Dataframe description are:", render_dict(df_list.describe().to_dict())),
    card("Shape of the Dataframe created by Lists is:", render_dict({"Shape": df_list.shape})),
    card("Dataframe created by Numpy Array is:", render_dataframe(df_array)),
    card("Type of the Dataframe created by Numpy Array is:", render_dict({"Type": type(df_array).__name__})),
    card("Dataframe description are:", render_dict(df_array.describe().to_dict())),
    card("Shape of the Dataframe created by Numpy Array is:", render_dict({"Shape": df_array.shape})),
    card("First 2 rows of the Dataframe created by Numpy Array:", render_dataframe(df_array_head)),
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
