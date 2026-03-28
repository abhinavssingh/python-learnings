import numpy as np
import pandas as pd
import io

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_dataframe, render_dict, render_pre
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running Pandas DataFrame Fundamentals report...")
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

# Creating a DataFrame from a NumPy array using random method

# For reproducibility (optional)
np.random.seed(42)

# Number of employees
n = 10

# Data generation
data = {
    "Name": [f"Employee_{i+1}" for i in range(n)],

    "Age": np.random.randint(22, 40, size=n),

    "Gender": np.random.choice(
        ["Male", "Female"], size=n
    ),

    "Salary": np.random.randint(
        40000, 100000, size=n
    ),

    "City": np.random.choice(
        ["Delhi", "Gurgaon", "Noida", "Faridabad", "Mohali", "Chandigarh", "Mumbai"], size=n
    ),

    "Zipcode": np.random.choice(
        [110001, 122001, 201301, 121001, 160055, 160001, 400001], size=n
    ),

    "Education": np.random.choice(
        ["B.Tech", "MBA", "B.Com", "M.Tech", "MCA", "BBA"], size=n
    ),

    "Passing Year": np.random.randint(
        2012, 2023, size=n
    )
}

# Create DataFrame
hr_df = pd.DataFrame(data)


# Create Employee ID as the DataFrame Index (Best Practice)
# hr_df.index = [f"EMP{1001+i}" for i in range(n)]
# hr_df.index.name = "Employee_ID"

# Keep Employee ID as a Column, Then Set Index
hr_df.insert(0, "Employee_ID", [f"EMP{1001+i}" for i in range(n)])
hr_df.set_index("Employee_ID", inplace=True)


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
    card("HR Dataframe created by Random Numpy Array is:", render_dataframe(hr_df)),
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
