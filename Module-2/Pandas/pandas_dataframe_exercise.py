import numpy as np
import pandas as pd

from lib.html import HtmlBuilder
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Pandas dataframe exercise  report...")
    # ...


df_basics = """
DataFrame axis is always 0 or 1
- Independent of DataFrame shape
- Shape size does not change axis values.
- Axis values are always 0 or 1 for a DataFrame, regardless of size.
- axis=0 → column-wise
- axis=1 → row-wise
- ❌ “Axis depends on number of rows/columns”
- ✅ “Axis depends on direction of operation”

✅ Key Rule (memorize this)
✅ Axis ALWAYS refers to the direction you collapse
- axis=0 → collapse rows → keep columns
- axis=1 → collapse columns → keep rows

------------------------------------------------
Axis | Direction    | Operation
------------------------------------------------
0    | Column-wise  | Collapse rows, keep columns
1    | Row-wise     | Collapse columns, keep rows
------------------------------------------------
"""

"""
## __Steps to Perform:__
- Determine the correlation between different numerical variables such as __LotArea__ and __SalePrice__
, __YearBuilt__ and __SalePrice__, __1stFlrSF__ and __SalePrice__, and __2ndFlrSF__ and __SalePrice__
"""
# Build full column-wise page

builder = HtmlBuilder()

df = dl.read_dataset("housing_data.csv", handle_unnamed="drop")
df[['YearBuilt', 'YearRemodAdd']] = df[['YearBuilt',
                                        'YearRemodAdd']].apply(pd.to_datetime, format='%Y')
diff = df["YearRemodAdd"].dt.year - df["YearBuilt"].dt.year

# Create category using vectorized logic
# Use cut when you need to segment and sort data values into bins.
# This function is also useful for going from a continuous variable to a categorical variable.
category_house = pd.cut(
    diff,
    bins=[-np.inf, 5, 20, 50, np.inf],
    labels=[
        "New Construction",
        "Modern Resale",
        "Established",
        "Historical"
    ],
)


# Insert the new column at the calculated location
df = dfh.insert_column_after(
    df, after_col="YearRemodAdd", new_col="HouseAge", values=diff, inplace=True)
df = dfh.insert_column_after(
    df, after_col="HouseAge", new_col="HouseCategory", values=category_house, inplace=True)
category_house_label_encoding = df['HouseCategory'].astype(
    'category').cat.codes
df = dfh.insert_column_after(df, after_col="HouseCategory", new_col="HouseCategoryEncoded",
                             values=category_house_label_encoding, inplace=True)
content = []

# Get information about the DataFrame
df_info_str = dfh.get_dataframe_info_str(df)

lot_area_desc = df["LotArea"].describe()
# YearBuilt is now datetime, so describe will give count, unique, top, freq instead of mean, std, etc.
year_built_desc = df["YearBuilt"].describe()
first_flr_desc = df["1stFlrSF"].describe()
second_flr_desc = df["2ndFlrSF"].describe()
sale_price_desc = df["SalePrice"].describe()

statistics_summary = {
    "LotArea Mean": lot_area_desc["mean"],
    "YearBuilt Mean": year_built_desc["mean"],
    "1stFlrSF Mean": first_flr_desc["mean"],
    "2ndFlrSF Mean": second_flr_desc["mean"],
    "SalePrice Mean": sale_price_desc["mean"],
    "LotArea Median": lot_area_desc["50%"],
    "YearBuilt Median": year_built_desc["50%"],
    "1stFlrSF Median": first_flr_desc["50%"],
    "2ndFlrSF Median": second_flr_desc["50%"],
    "SalePrice Median": sale_price_desc["50%"],
    "LotArea Std Dev": lot_area_desc["std"],
    # "YearBuilt Std Dev": year_built_desc["std"], # YearBuilt is datetime, so std is not applicable
    "1stFlrSF Std Dev": first_flr_desc["std"],
    "2ndFlrSF Std Dev": second_flr_desc["std"],
    "SalePrice Std Dev": sale_price_desc["std"]
}

corelation_summary = {
    "LotArea vs SalePrice": df["LotArea"].corr(df["SalePrice"]),
    "YearBuilt vs SalePrice": df["YearBuilt"].corr(df["SalePrice"]),
    "1stFlrSF vs SalePrice": df["1stFlrSF"].corr(df["SalePrice"]),
    "2ndFlrSF vs SalePrice": df["2ndFlrSF"].corr(df["SalePrice"])
}

category_summary = {
    "Unique Neighborhoods": df["Neighborhood"].value_counts().sort_values(ascending=False).to_dict(),
    "Unique Building Types": df["BldgType"].value_counts().sort_values(ascending=False).to_dict(),
    "Unique House Styles": df["HouseStyle"].value_counts().sort_values(ascending=False).to_dict()
}

category_counts = {
    "Neighborhood": df.groupby("Neighborhood").size().to_dict(),
    "BldgType": df.groupby("BldgType").size().to_dict(),
    "HouseStyle": df.groupby("HouseStyle").size().to_dict()
}

house_category_counts = {
    "HouseCategory": df.groupby("HouseCategory").size().to_dict(),
    "HouseCategoryEncoded": df.groupby("HouseCategoryEncoded").size().to_dict()
}

numeric_df = df.select_dtypes(include=["number"])
non_numeric_df = df.select_dtypes(exclude=["number"])

# use for the large dataset
content.append(
    builder.full_width_card(
        "Housing Dataset – Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(builder.full_width_card(
    "Numeric DataFrame",
    builder.render_dataframe_collapsible(numeric_df, initial_rows=10)
))

content.append(builder.full_width_card(
    "Non-Numeric DataFrame",
    builder.render_dataframe_collapsible(non_numeric_df, initial_rows=10)
))

content.append(builder.full_width_card(
    "Statistics of the Numeric DataFrame Using Transposed",
    builder.render_dataframe_collapsible(
        numeric_df.describe().T, initial_rows=10)
))

# DataFrame.describe() is dtype‑aware
# By default:
# It summarizes numeric columns
# If no numeric columns, it summarizes datetime columns
# It ignores plain object (string/categorical) columns unless explicitly told otherwise
content.append(builder.full_width_card(
    "Statistics of the Non-Numeric DataFrame Using Transposed",
    builder.render_dataframe_collapsible(
        non_numeric_df.describe(include="all").T, initial_rows=10)
))

# Section — small summaries
content.append(
    builder.grid([
        builder.card("DataFrame Basics:", builder.render_pre(df_basics)),
        builder.card("Information of the Housing Dataframe is:",
                     builder.render_pre(df_info_str)),
        builder.card("Shape of the Housing DataFrame is:",
                     builder.render_dict({"Shape": df.shape})),
        builder.card("Dataframe description are:",
                     builder.render_dict(df.describe().to_dict())),
        builder.card("Unique Neighborhoods, Building Types, and House Styles using Category and sorting in descending order:",
                     builder.render_dict(category_summary)),
        builder.card("Unique Neighborhoods, Building Types, and House Styles Counts using groupby:",
                     builder.render_dict(category_counts)),
        builder.card("Lot Area Description:",
                     builder.render_dict(lot_area_desc.to_dict())),
        builder.card("Year Built Description:",
                     builder.render_dict(year_built_desc.to_dict())),
        builder.card("1st Floor SF Description:",
                     builder.render_dict(first_flr_desc.to_dict())),
        builder.card("2nd Floor SF Description:",
                     builder.render_dict(second_flr_desc.to_dict())),
        builder.card("Sale Price Description:",
                     builder.render_dict(sale_price_desc.to_dict())),
        builder.card(" Statistics of the Numeric DataFrame:",
                     builder.render_dict(numeric_df.describe().to_dict())),
        builder.card("Statistics of the Non-Numeric DataFrame:",
                     builder.render_dict(non_numeric_df.describe().to_dict())),
        builder.card("Summary of Descriptive Statistics for Key Numerical Columns:",
                     builder.render_dict(statistics_summary)),
        builder.card("Correlation Summary for Key Numerical Columns:",
                     builder.render_dict(corelation_summary)),
        builder.card("House Category Counts:",
                     builder. render_dict(house_category_counts))
    ])
)


html_doc = builder.build_page(
    "Pandas Exercise  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "pandas_dataframe_exercise_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
