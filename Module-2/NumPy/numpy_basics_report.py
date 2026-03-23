import numpy as np
from lib.arrays_html import arrays_report_html
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy basics report...")
    # ...


# Your arrays
arr1 = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8]])

arr1_reshape = arr1.reshape(4, 2)
arr1_t = arr1.transpose()

arr2 = np.array([[1., 2., 3., 4.],
                 [5., 6., 7., 8.],
                 [9., 10., 11., 12.]])
arr2_reshape = arr2.reshape(2, 3, 2)
arr2_t = arr2.transpose()

arr3 = np.array([['1', '2', '3', '4'],
                 ['5', '6', '7', '8']], dtype=str)
arr3_reshape = arr3.reshape(4, 2)
arr3_t = arr3.transpose()

# Build full column-wise page
html_doc = arrays_report_html([
    ("Original array (int)", arr1),
    ("Reshape array (int)", arr1_reshape),
    ("Transpose array (int)", arr1_t),
    ("Original array (float)", arr2),
    ("Reshape array (float)", arr2_reshape),
    ("Transpose array (float)", arr2_t),
    ("Original array (str)", arr3),
    ("Reshape array (str)", arr3_reshape),
    ("Transpose array (str)", arr3_t),
], page_title="Array Details (Column-wise)")


# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "arrays_basics_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
