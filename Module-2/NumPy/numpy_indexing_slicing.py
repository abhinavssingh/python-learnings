import numpy as np

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_array, render_kv
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy Indexing and Slicing report...")

    # ...


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

html_doc = build_html_page(
    "NumPy Indexing and Slicing Examples",
    grid([
        card("Original 1D array", render_array(array_1d)),
        card("value at index 3 of the 1D NumPy array", render_kv([("Value", array_1d[3])])),
        card("sum of the values at indexes 0 and 1 in the 1D NumPy array", render_kv([("Sum", array_1d[1] + array_1d[0])])),
        card("fourth element from the end of the 1D array using negative indexing", render_kv([("Value", array_1d[-4])])),
        card("Slicing on 1D array based on negative pattern [-2:-8:-3]", render_array(array_1d[-2:-8:-3])),
        card("Slicing on 1D array based on pattern [2:10:2]", render_array(array_1d[2:10:2])),
        card("Negative slicing for 1D array based on pattern [:-1]", render_array(array_1d[:-1])),
        card("Original 2D array", render_array(array_2d)),
        card('Third element in the first row of the 2D array: ', render_kv([("Value", array_2d[0, 2])])),
        card('Second element in the second row of the 2D array: ', render_kv([("Value", array_2d[1, 1])])),
        card('Slicing on 2D array based on pattern [1, 1:4]: ', render_array(array_2d[1, 1:4])),
        card('Slicing on 2D array based on pattern [0, 1:5:2]: ', render_array(array_2d[0, 1:5:2])),
        card('Negative Slicing on 2D array based on pattern [2, -1:-5:-2]: ', render_array(array_2d[2, -1:-5:-2])),
        card("last element in the second row of the 2D array using negative indexing", render_array(array_2d[1, -1])),
        card("Slicing on 2D array based on pattern [0, 2:3] ", render_array(array_2d[0, 2:3])),
        card("Original 3D array", render_array(array_3d)),
        card("first element of the first row of the 3D array", render_kv([("Value", array_3d[1, 0, 0])])),
        card("last element in the last row of the 3D array using negative indexing [1, 1, -1]", render_kv([("Value", array_3d[1, 1, -1])])),
        card("Middle element in the first row of the 3D array using negative indexing [1, 0, -2]", render_kv([("Value", array_3d[1, 0, -2])])),
        card("Middle element in the second row of the 3D array using negative indexing [1, 1, -2]", render_kv([("Value", array_3d[1, 1, -2])])),
        card("Slicing on 3D array based on pattern [0,1:,1:]", render_array(array_3d[0, 1:, 1:])),
        card("Negative Slicing on 3D array based on pattern [0,0:,-1:]", render_array(array_3d[0, 0:, -1::])),
        card("Negative Slicing on 3D array based on pattern [0,0:,-1::]", render_array(array_3d[0, 0:, -1:])),
        card("Negative Slicing on 3D array based on pattern [1,1:,-2:]", render_array(array_3d[1, 1:, -2:])),
    ])
)

# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "numpy_indexing_slicing_report.html",
    html_doc,
    subfolder="reports",
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
