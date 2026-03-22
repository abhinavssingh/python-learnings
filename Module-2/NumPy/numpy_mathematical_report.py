
from pathlib import Path
import sys

# Add the parent of the current folder (…\Module-2) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from lib.arrays_html import arrays_report_html
import webbrowser, os

# Your arrays
arr1 = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

arr2 = np.array(
    [
        [9,10,11,12],
        [13,14,15,16]
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
sum = np.add(arr1,arr2)
sub = np.subtract(arr2,arr1)
multiplication = np.multiply(arr1,arr2)
division = np.divide(arr1,arr2)
rounded_division = np.round(division,3)
power = np.power(arr1,arr2)

sum_3d = np.add(a3,array_3d)
sub_3d = np.subtract(a3,array_3d)
multiplication_3d = np.multiply(a3,array_3d)
division_3d = np.divide(a3,array_3d).round(3)
power_3d = np.power(a3,array_3d)

#statistical operations
arr1_mean = np.mean(arr1)
arr1_median = np.median(arr1)
arr1_std = np.std(arr1).round(3)
arr1_var = np.var(arr1)
arr1_per_80 = np.percentile( arr1, 80)
arr1_per_90 = np.percentile( arr1, 90)

arr3d_mean = np.mean(array_3d)
arr3d_median = np.median(array_3d)
arr3d_std = np.std(array_3d).round(3)
arr3d_var = np.var(array_3d)
arr3d_per_80 = np.percentile( array_3d, 80)
arr3d_per_90 = np.percentile( array_3d, 90)

# Build full column-wise page
html_doc = arrays_report_html([
    ("Original 1st 2D array (int)",   arr1),
    ("Original 2nd 2D array (int)",   arr2),
    ("Sum of two array (int)",   sum),
    ("Subtract of two array array2-array1 (int)",   sub),
    ("Multiplication of two array (int)", multiplication),
    ("Division of two array array/array2 (int)",   division),
    ("Division of two array array/array2 after rounding (int)",   rounded_division),
    ("Power of two array (int)",   power),
    ("Mean of array 1 (int)",   arr1_mean),
    ("Median of array 1 (int)",   arr1_median),
    ("Standard Deviation of array 1 (int)",   arr1_std),
    ("Variance of array 1 (int)",   arr1_var),
    ("80 Percentile of array 1 (int)",   arr1_per_80),
    ("90 Percentile of array 1 (int)",   arr1_per_90),
    ("Original 1st 3D array (int)",   a3),
    ("Original 2nd 3D array (int)",   array_3d),
    ("Sum of two 3D array (int)",   sum_3d),
    ("Subtract of two 3D array array2-array1 (int)",   sub_3d),
    ("Multiplication of two 3D array (int)", multiplication_3d),
    ("Division of two 3D array array/array2 (int)",   division_3d),
    ("Power of two 3D array (int)",   power_3d),
    ("Mean of 1st 3D array (int)",   arr3d_mean),
    ("Median of 1st 3D array 1 (int)",   arr3d_median),
    ("Standard Deviation of 1st 3D array 1 (int)",   arr3d_std),
    ("Variance of 1st 3D array (int)",   arr3d_var),
    ("80 Percentile of 1st 3D array (int)",   arr3d_per_80),
    ("90 Percentile of 1st 3D array (int)",   arr3d_per_90),
], page_title="Array Details (Column-wise)")

# Save
with open("arrays_maths_report.html", "w", encoding="utf-8") as f:
    f.write(html_doc)

print("Wrote arrays_maths_report.html")


webbrowser.open('file://' + os.path.abspath('arrays_maths_report.html'))