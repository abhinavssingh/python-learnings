import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Pandas DataFrame Fundamentals report...")
    # ...

# Build full column-wise page


builder = HtmlBuilder()

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
    "Name": [f"Employee_{i + 1}" for i in range(n)],

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
hr_df.insert(0, "Employee_ID", [f"EMP{1001 + i}" for i in range(n)])
hr_df.set_index("Employee_ID", inplace=True)
hr_emp_data_str = dfh.dataframe_rows_as_pre(
    hr_df, method="iterrows", include_index=True, index_label="Employee ID")
hr_emp_data_str_salary = dfh.dataframe_rows_as_pre(
    hr_df.sort_values(by="Salary", ascending=False))

df_array = pd.DataFrame(data_array, columns=columns_name)
df_array_head = df_array.head(2)  # Get the first few rows of the DataFrame
df_dict_tail = df_dict.tail(1)  # Get the last few rows of the DataFrame
# Get information about the DataFrame
df_dict_info_str = dfh.get_dataframe_info_str(df_dict)

html_doc = builder.build_page("Pandas Dataframe Fundamentals Report", builder.grid([
    builder.card("Dataframe created by Dictionary is:",
                 builder.render_dataframe(df_dict)),
    builder.card("Type of the Dataframe created by Dictionary is:",
                 builder.render_dict({"Type": type(df_dict).__name__})),
    builder.card("Dataframe description are:",
                 builder.render_dict(df_dict.describe().to_dict())),
    builder.card("Information about the Dataframe created by Dictionary is:",
                 builder.render_pre(df_dict_info_str)),
    builder.card("Last row of the Dataframe created by Dictionary:",
                 builder.render_dataframe(df_dict_tail)),
    builder.card("Shape of the Dataframe is:",
                 builder.render_dict({"Shape": df_dict.shape})),
    builder.card("Dataframe created by Lists is:",
                 builder.render_dataframe(df_list)),
    builder.card("Type of the Dataframe created by Lists is:",
                 builder.render_dict({"Type": type(df_list).__name__})),
    builder.card("Dataframe description are:",
                 builder.render_dict(df_list.describe().to_dict())),
    builder.card("Shape of the Dataframe created by Lists is:",
                 builder.render_dict({"Shape": df_list.shape})),
    builder.card("Dataframe created by Numpy Array is:",
                 builder.render_dataframe(df_array)),
    builder.card("Type of the Dataframe created by Numpy Array is:",
                 builder.render_dict({"Type": type(df_array).__name__})),
    builder.card("Dataframe description are:",
                 builder.render_dict(df_array.describe().to_dict())),
    builder.card("Shape of the Dataframe created by Numpy Array is:",
                 builder.render_dict({"Shape": df_array.shape})),
    builder.card("First 2 rows of the Dataframe created by Numpy Array:",
                 builder.render_dataframe(df_array_head)),
    builder.card("HR Dataframe created by Random Numpy Array is:",
                 builder.render_dataframe(hr_df)),
    builder.card("HR Dataframe description are:",
                 builder.render_dict(hr_df.describe().to_dict())),
    builder.card("HR Dataframe Employee Data is:",
                 builder.render_pre(hr_emp_data_str)),
    builder.card("HR Dataframe Employee Salary Data sorted by Salary in Descending Order is:",
                 builder.render_pre(hr_emp_data_str_salary)),
]))


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "pandas_dataframe_fundamentals_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
