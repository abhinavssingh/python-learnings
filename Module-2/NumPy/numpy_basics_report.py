import numpy as np

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


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

builder = HtmlBuilder()

html = builder.build_page(
    "NumPy Basic Report",
    builder.grid([
        builder.card("Original Array (int)", builder.render_array(arr1)),
        builder.card("Reshape Array (int)",
                     builder.render_array(arr1_reshape)),
        builder.card("Transpose Array (int)", builder.render_array(arr1_t)),
        builder.card("Original Array (float)", builder.render_array(arr2)),
        builder.card("Reshape Array (float)",
                     builder.render_array(arr2_reshape)),
        builder.card("Transpose Array (float)", builder.render_array(arr2_t)),
        builder.card("Original Array (str)", builder.render_array(arr3)),
        builder.card("Reshape Array (str)",
                     builder.render_array(arr3_reshape)),
        builder.card("Transpose Array (str)", builder.render_array(arr3_t)),
    ])
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "numpy_basics_report.html",   # file name
    html,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
