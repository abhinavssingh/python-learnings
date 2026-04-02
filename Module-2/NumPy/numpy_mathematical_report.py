import numpy as np

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running NumPy Mathematical Operations report...")
    # ...

# Build full column-wise page


builder = HtmlBuilder()

# Your arrays
arr1 = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

arr2 = np.array(
    [
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

array_3d = np.array(
    [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [7, 8, 9],
            [10, 11, 12]
        ]
    ])

a3 = np.array([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])


# mathematical operations
sum = np.add(arr1, arr2)
sub = np.subtract(arr2, arr1)
multiplication = np.multiply(arr1, arr2)
division = np.divide(arr1, arr2)
rounded_division = np.round(division, 3)
power = np.power(arr1, arr2)

sum_3d = np.add(a3, array_3d)
sub_3d = np.subtract(a3, array_3d)
multiplication_3d = np.multiply(a3, array_3d)
division_3d = np.divide(a3, array_3d).round(3)
power_3d = np.power(a3, array_3d)

# statistical operations
arr1_mean = np.mean(arr1)
arr1_median = np.median(arr1)
arr1_std = np.std(arr1).round(3)
arr1_var = np.var(arr1)
arr1_per_80 = np.percentile(arr1, 80)
arr1_per_90 = np.percentile(arr1, 90)

arr3d_mean = np.mean(array_3d)
arr3d_median = np.median(array_3d)
arr3d_std = np.std(array_3d).round(3)
arr3d_var = np.var(array_3d)
arr3d_per_80 = np.percentile(array_3d, 80)
arr3d_per_90 = np.percentile(array_3d, 90)

# Build full column-wise page
html_doc = builder.build_page(
    "NumPy Mathematical Operations Report",
    builder.grid([
        builder.card("Original 1st 2D array (int)",
                     builder.render_array(arr1)),
        builder.card("Original 2nd 2D array (int)",
                     builder.render_array(arr2)),
        builder.card("Sum of two array (int)", builder.render_array(sum)),
        builder.card("Subtract of two array array2-array1 (int)",
                     builder.render_array(sub)),
        builder.card("Multiplication of two array (int)",
                     builder.render_array(multiplication)),
        builder.card("Division of two array array/array2 (int)",
                     builder.render_array(division)),
        builder.card("Division of two array array/array2 after rounding (int)",
                     builder.render_array(rounded_division)),
        builder.card("Power of two array (int)", builder.render_array(power)),
        builder.card("Mean of array 1 (int)",
                     builder.render_dict({"Mean": arr1_mean})),
        builder.card("Median of array 1 (int)",
                     builder.render_dict({"Median": arr1_median})),
        builder.card("Standard Deviation of array 1 (int)",
                     builder.render_dict({"Standard Deviation": arr1_std})),
        builder.card("Variance of array 1 (int)",
                     builder.render_dict({"Variance": arr1_var})),
        builder.card("80 Percentile of array 1 (int)",
                     builder.render_dict({"80 Percentile": arr1_per_80})),
        builder.card("90 Percentile of array 1 (int)",
                     builder.render_dict({"90 Percentile": arr1_per_90})),
        builder.card("Original 1st 3D array (int)", builder.render_array(a3)),
        builder.card("Original 2nd 3D array (int)",
                     builder.render_array(array_3d)),
        builder.card("Sum of two 3D array (int)",
                     builder.render_array(sum_3d)),
        builder.card("Subtract of two 3D array array2-array1 (int)",
                     builder.render_array(sub_3d)),
        builder.card("Multiplication of two 3D array (int)",
                     builder.render_array(multiplication_3d)),
        builder.card("Division of two 3D array array/array2 (int)",
                     builder.render_array(division_3d)),
        builder.card("Power of two 3D array (int)",
                     builder.render_array(power_3d)),
        builder.card("Mean of 1st 3D array (int)",
                     builder.render_dict({"Mean": arr3d_mean})),
        builder.card("Median of 1st 3D array 1 (int)",
                     builder.render_dict({"Median": arr3d_median})),
        builder.card("Standard Deviation of 1st 3D array 1 (int)",
                     builder.render_dict({"Standard Deviation": arr3d_std})),
        builder.card("Variance of 1st 3D array (int)",
                     builder.render_dict({"Variance": arr3d_var})),
        builder.card("80 Percentile of 1st 3D array (int)",
                     builder.render_dict({"80 Percentile": arr3d_per_80})),
        builder.card("90 Percentile of 1st 3D array (int)",
                     builder.render_dict({"90 Percentile": arr3d_per_90})),
    ])
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "numpy_mathematical_report.html",
    html_doc,
    subfolder="reports",
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
