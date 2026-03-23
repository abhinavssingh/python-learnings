import pandas as pd
import numpy as np
from lib.arrays_html import arrays_table_html, arrays_index_report_html
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
    # ...


sales_data = [120, 150, 130, 170, 160, 180, 140]
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sales_series = pd.Series(sales_data, index=days_of_week)
sales_series["Sunday"] = 300
sales_series_preferred_days = sales_series.loc[['Monday', 'Wednesday', 'Friday', 'Sunday']]
sales_series_on_contains = sales_series[sales_series.index.str.contains("ur") | sales_series.index.str.contains("ed")]
sales_series_total = sales_series.sum()
sales_series_average = sales_series.mean()
sales_serries_max = sales_series.max()
sales_serries_max_label = sales_series.idxmax()
sales_series_min = sales_series.min()
sales_series_min_label = sales_series.idxmin()

# calculate the sale which is far from average, we tell outlier in data sciencr
# am following statistical approach rather than thresold
sales_series_std = sales_series.std()
q1 = sales_series.quantile(0.25)
q3 = sales_series.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

# IQR outliers
outliers_iqr = sales_series[(sales_series.astype(float) < lower_fence) | (sales_series.astype(float) > upper_fence)]

# Z-scores (population vs sample doesn't matter for ranking); using sample std
z = ((sales_series - sales_series_average) / sales_series_std).round(3)
outliers_z = z[(z > 2.0) | (z < -2.0)]
common_index = sales_series.index.intersection(outliers_z.index)
outliers_z_actual_values = sales_series[common_index]


# MAD / robust z
median = sales_series.median()
mad = (sales_series - median).abs().median()
robust_z = 0.6745 * (sales_series - median) / (mad if mad != 0 else np.nan)  # avoid div 0

# Day-over-day changes
pct_change = sales_series.pct_change() * 100
abs_change = sales_series.diff()

# Ranks
rank_desc = sales_series.rank(ascending=False, method='min')

pairs = [
    (" Sales Series is:", sales_series),
    (" Sales Serris Describe bare:", sales_series.describe().to_dict()),
    (" Type of the Sales Series is:", sales_series.dtype),
    (" Sales Series based on labels:", sales_series_preferred_days),
    (" Sales Series based on contains:", sales_series_on_contains),
    (" Sales Series based on label contains ur or ed:", sales_series_on_contains),
    (" Total Sales:", sales_series_total),
    (" Average Sales:", sales_series_average),
    (" Maximum Sales is occured on:", sales_serries_max_label),
    (" Maximum Sale value is:", sales_serries_max),
    (" Minimum Sales is occured on:", sales_series_min_label),
    (" Minimum Sale value is:", sales_series_min),
    (" Standard Deviation is:", sales_series_std),
    (" Q1 is:", q1),
    ("Q3 is:", q3),
    (" IQR is:", iqr),
    (" Lower value as per IQR is:", lower_fence),
    (" Maximum value as per IQR is:", upper_fence),
    (" Outlier or the element which is far from average is:", outliers_iqr),
    (" Recommendation is to drop the outlier element for better decision", outliers_iqr.values),
    (" Z score is:", z),
    (" Acatual values in outliers based on Z score:", outliers_z_actual_values),
    (" As per sigma 2 rule outlier is:", outliers_z),
    (" Robust Z score is:", robust_z),
    (" Rank is:", rank_desc)
]

# 1) Just the fragment (embed in an existing page or notebook cell)
fragment = arrays_table_html(pairs)

# 2) Full standalone page
html_doc = arrays_index_report_html(pairs, page_title="Pandas Series Exercise Report")

# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "pandas_series_exercise_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
