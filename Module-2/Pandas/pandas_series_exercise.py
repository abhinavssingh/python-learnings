import numpy as np
import pandas as pd

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_series, render_dict
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running Pandas series exercise report...")
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

results = {
    "Total Sales": sales_series_total,
    "Average Sales": sales_series_average,
    "Maximum Sales": sales_serries_max,
    "Day of Maximum Sales": sales_serries_max_label,
    "Minimum Sales": sales_series_min,
    "Day of Minimum Sales": sales_series_min_label,
    "Standard Deviation": sales_series_std,
    "Q1": q1,
    "Q3": q3,
    "IQR": iqr,
    "Lower Fence": lower_fence,
    "Upper Fence": upper_fence,
    "Outliers (IQR)": outliers_iqr.tolist(),
    "Z Scores": z.round(3).tolist(),
    "Outliers (Z-Score)": outliers_z.round(3).tolist(),
    "Outliers (Z-Score) - Actual Values": outliers_z_actual_values.tolist(),
    "Robust Z Scores": robust_z.round(3).tolist(),
    "Rank (Descending)": rank_desc.astype(int).tolist()
}

html_doc = build_html_page("Pandas Series Exercise Report", grid(
    [
        card(" Sales Series is:", render_series(sales_series)),
        card(" Sales Series Describe bare:", render_dict({"Description": sales_series.describe().to_dict()})),
        card(" Type of the Sales Series is:", render_dict({"Type": type(sales_series).__name__})),
        card(" Sales Series based on labels:", render_series(sales_series_preferred_days)),
        card(" Sales Series based on contains:", render_series(sales_series_on_contains)),
        card(" Sales Series based on label contains ur or ed:", render_series(sales_series_on_contains)),
        card(" Summary of the Sales Series is:", render_dict({"Summary": results})),
        card(" Outlier or the element which is far from average is:", render_series(outliers_iqr)),
        card(" Recommendation is to drop the outlier element for better decision", render_series(outliers_iqr.values)),
        card(" Z score is:", render_series(z)),
        card(" Acatual values in outliers based on Z score:", render_series(outliers_z_actual_values)),
        card(" As per sigma 2 rule outlier is:", render_series(outliers_z)),
    ]))


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
