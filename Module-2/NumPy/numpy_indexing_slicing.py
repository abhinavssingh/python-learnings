import numpy as np

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running NumPy Indexing and Slicing report...")

    # ...

# Build full column-wise page


builder = HtmlBuilder()

array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
array_2d = np.array(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]
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

html_doc = builder.build_page(
    "NumPy Indexing and Slicing Examples",
    builder.grid([
        builder.card("Original 1D array", builder.render_array(array_1d)),
        builder.card("value at index 3 of the 1D NumPy array", builder.render_kv([("Value", array_1d[3])])),
        builder.card("sum of the values at indexes 0 and 1 in the 1D NumPy array", builder.render_kv([("Sum", array_1d[1] + array_1d[0])])),
        builder.card("fourth element from the end of the 1D array using negative indexing", builder.render_kv([("Value", array_1d[-4])])),
        builder.card("Slicing on 1D array based on negative pattern [-2:-8:-3]", builder.render_array(array_1d[-2:-8:-3])),
        builder.card("Slicing on 1D array based on pattern [2:10:2]", builder.render_array(array_1d[2:10:2])),
        builder.card("Negative slicing for 1D array based on pattern [:-1]", builder.render_array(array_1d[:-1])),
        builder.card("Original 2D array", builder.render_array(array_2d)),
        builder.card('Third element in the first row of the 2D array: ', builder.render_kv([("Value", array_2d[0, 2])])),
        builder.card('Second element in the second row of the 2D array: ', builder.render_kv([("Value", array_2d[1, 1])])),
        builder.card('Negative Slicing on 2D array based on pattern [2, -1:-5:-2]: ', builder.render_array(array_2d[2, -1:-5:-2])),
        builder.card("last element in the second row of the 2D array using negative indexing", builder.render_array(array_2d[1, -1])),
        builder.card("Slicing on 2D array based on pattern [0, 2:3] ", builder.render_array(array_2d[0, 2:3])),
        builder.card("Original 3D array", builder.render_array(array_3d)),
        builder.card("first element of the first row of the 3D array", builder.render_kv([("Value", array_3d[1, 0, 0])])),
        builder.card(
            "last element in the last row of the 3D array using negative indexing [1, 1, -1]", builder.render_kv([("Value", array_3d[1, 1, -1])])),
        builder.card(
            "Middle element in the first row of the 3D array using negative indexing [1, 0, -2]", builder.render_kv([("Value", array_3d[1, 0, -2])])),
        builder.card(
            "Middle element in the second row of the 3D array using negative indexing [1, 1, -2]", builder.render_kv([("Value", array_3d[1, 1, -2])])),
        builder.card("Slicing on 3D array based on pattern [0,1:,1:]", builder.render_array(array_3d[0, 1:, 1:])),
        builder.card("Negative Slicing on 3D array based on pattern [0,0:,-1:]", builder.render_array(array_3d[0, 0:, -1::])),
        builder.card("Negative Slicing on 3D array based on pattern [0,0:,-1::]", builder.render_array(array_3d[0, 0:, -1:])),
        builder.card("Negative Slicing on 3D array based on pattern [1,1:,-2:]", builder.render_array(array_3d[1, 1:, -2:])),
    ])
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "numpy_indexing_slicing_report.html",
    html_doc,
    subfolder="reports",
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
