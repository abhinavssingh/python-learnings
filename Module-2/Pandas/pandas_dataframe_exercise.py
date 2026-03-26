from lib.data_loader import read_dataset
import io
from lib.html.base import build_html_page
from lib.html.components import full_width_card, grid, card
from lib.html.renderers import render_dataframe_collapsible, render_pre, render_dict
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
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
df = read_dataset("housing_data.csv")

content = []

# Get information about the DataFrame
buffer = io.StringIO()
df_info = df.info(buf=buffer)
df_info_str = buffer.getvalue()  # Retrieve the string from the buffer

lot_area_desc = df["LotArea"].describe()
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
    "YearBuilt Std Dev": year_built_desc["std"],
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
    "Unique Neighborhoods":  df["Neighborhood"].value_counts().sort_values(ascending=False).to_dict(),
    "Unique Building Types": df["BldgType"].value_counts().sort_values(ascending=False).to_dict(),
    "Unique House Styles": df["HouseStyle"].value_counts().sort_values(ascending=False).to_dict()
}

category_counts = {
    "Neighborhood": df.groupby("Neighborhood").size().to_dict(),
    "BldgType": df.groupby("BldgType").size().to_dict(),
    "HouseStyle": df.groupby("HouseStyle").size().to_dict()
}

numeric_df = df.select_dtypes(include=["number"])
non_numeric_df = df.select_dtypes(exclude=["number"])

# use for the large dataset
content.append(
    full_width_card(
        "Housing Dataset – Interactive Preview",
        render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(full_width_card(
    "Numeric DataFrame",
    render_dataframe_collapsible(numeric_df, initial_rows=10)
))

content.append(full_width_card(
    "Non-Numeric DataFrame",
    render_dataframe_collapsible(non_numeric_df, initial_rows=10)
))

content.append(full_width_card(
    "Statistics of the Numeric DataFrame Using Transposed",
    render_dataframe_collapsible(numeric_df.describe().T, initial_rows=10)
))

content.append(full_width_card(
    "Statistics of the Non-Numeric DataFrame Using Transposed",
    render_dataframe_collapsible(non_numeric_df.describe().T, initial_rows=10)
))

# Section — small summaries
content.append(
    grid([
        card("DataFrame Basics:", render_pre(df_basics)),
        card("Information of the Housing Dataframe is:", render_pre(df_info_str)),
        card("Shape of the Housing DataFrame is:", render_dict({"Shape": df.shape})),
        card("Dataframe description are:", render_dict(df.describe().to_dict())),
        card("Unique Neighborhoods, Building Types, and House Styles using Category and sorting in descending order:", render_dict(category_summary)),
        card("Unique Neighborhoods, Building Types, and House Styles Counts using groupby:", render_dict(category_counts)),
        card("Lot Area Description:", render_dict(lot_area_desc.to_dict())),
        card("Year Built Description:", render_dict(year_built_desc.to_dict())),
        card("1st Floor SF Description:", render_dict(first_flr_desc.to_dict())),
        card("2nd Floor SF Description:", render_dict(second_flr_desc.to_dict())),
        card("Sale Price Description:", render_dict(sale_price_desc.to_dict())),
        card(" Statistics of the Numeric DataFrame:", render_dict(numeric_df.describe().to_dict())),
        card("Statistics of the Non-Numeric DataFrame:", render_dict(non_numeric_df.describe().to_dict())),
        card("Summary of Descriptive Statistics for Key Numerical Columns:", render_dict(statistics_summary)),
        card("Correlation Summary for Key Numerical Columns:", render_dict(corelation_summary))
    ])
)


html_doc = build_html_page(
    "Pandas Exercise  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_dataframe_exercise_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
