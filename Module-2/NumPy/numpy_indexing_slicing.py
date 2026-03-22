import numpy as np
import webbrowser, os
from lib.arrays_html import arrays_table_html, arrays_index_report_html

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

pairs = [
    ("Original 1D array", array_1d),
    ("value at index 3 of the 1D NumPy array", array_1d[3]),
    ("sum of the values at indexes 0 and 1 in the 1D NumPy array", array_1d[1] + array_1d[0]),
    ("fourth element from the end of the 1D array using negative indexing", array_1d[-4]),
    ("Slicing on 1D array based on negative pattern [-2:-8:-3]", array_1d[-2:-8:-3] ),
    ("Slicing on 1D array based on pattern [2:10:2]", array_1d[2:10:2] ),
    ("Negative slicing for 1D array based on pattern [:-1]", array_1d[:-1]),
    ("Original 2D array", array_2d),
    ('Third element in the first row of the 2D array: ', array_2d[0, 2]),
    ('Second element in the second row of the 2D array: ', array_2d[1, 1]),
    ('Slicing on 2D array based on pattern [1, 1:4]: ', array_2d[1, 1:4]),
    ('Slicing on 2D array based on pattern [0, 1:5:2]: ', array_2d[0, 1:5:2]),
    ('Negative Slicing on 2D array based on pattern [2, -1:-5:-2]: ', array_2d[2, -1:-5:-2]),
    ("last element in the second row of the 2D array using negative indexing", array_2d[1, -1]),
    ("Slicing on 2D array based on pattern [0, 2:3] ", array_2d[0, 2:3]),
    ("Original 3D array", array_3d),
    ("first element of the first row of the second 2D array within the 3D array", array_3d[1, 0, 0]),
    ("last element in the last row of the last 2D array within the 3D array using negative indexing [1, 1, -1]", array_3d[1, 1, -1]),
    ("Middle element in the first row of the last 2D array within the 3D array using negative indexing [1, 0, -2]", array_3d[1, 0, -2]),
    ("Middle element in the second row of the last 2D array within the 3D array using negative indexing [1, 1, -2]", array_3d[1, 1, -2]),
    ("Slicing on 3D array based on pattern [0,1:,1:]", array_3d[0,1:,1:]),
    ("Negative Slicing on 3D array based on pattern [0,0:,-1:]", array_3d[0,0:,-1::]),
    ("Negative Slicing on 3D array based on pattern [0,0:,-1::]", array_3d[0,0:,-1:]),
    ("Negative Slicing on 3D array based on pattern [1,1:,-2:]", array_3d[1,1:,-2:]),
]


# 1) Just the fragment (embed in an existing page or notebook cell)
fragment = arrays_table_html(pairs)

# 2) Full standalone page
doc = arrays_index_report_html(pairs, page_title="Arrays Indexing and Slicing Report")
with open("arrays_indexing_slicing_report.html", "w", encoding="utf-8") as f:
    f.write(doc)

print("Wrote arrays_indexing_slicing_report.html")


webbrowser.open('file://' + os.path.abspath('arrays_indexing_slicing_report.html'))
