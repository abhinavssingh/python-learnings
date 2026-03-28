import numpy as np

from lib.html.base import build_html_page
from lib.html.components import card, grid
from lib.html.renderers import render_array
from lib.report_utils import save_html_report


def main():
    # your current script code goes here
    print("Running NumPy Transpose Operations report...")
    # ...


# Your arrays
# 0D array
a0 = np.array(42)

# 1D array
a1 = np.array([1, 2, 3, 4])

# 2D array
a2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
a2_t = a2.transpose()

# Index-based transpose
# For 2D, a.T, np.transpose(a), and np.transpose(a, (1,0)) are equivalent.
a2_t_idx = a2.transpose(1, 0)

# 3D array
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
a3_t = a3.transpose()

# index based transpose
a3_t_021 = a3.transpose(0, 2, 1)
a3_t_120 = a3.transpose(1, 2, 0)
a3_t_102 = a3.transpose(1, 0, 2)

# 4D array
# Let’s define a small (N, C, H, W) = (2, 2, 2, 2) sample.
a4 = np.array([
    [
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ],
        [
            [9, 10],
            [11, 12]
        ]
    ],
    [
        [
            [13, 14],
            [15, 16]
        ],
        [
            [17, 18],
            [19, 20]
        ],
        [
            [21, 22],
            [23, 24]
        ]
    ]
])
a4_t = a4.transpose()

# index based transpose
a4_t_0231 = a4.transpose(0, 2, 3, 1)
a4_t_0312 = a4.transpose(0, 3, 1, 2)
a4_t_0132 = a4.transpose(0, 1, 3, 2)
a4_t_1023 = a4.transpose(1, 0, 2, 3)

# Build full column-wise page
html_doc = build_html_page(
    "NumPy Transpose Operations Report",
    grid([
        card("Original array 0D (int)", render_array(a0)),
        card("Original array 1D (int)", render_array(a1)),
        card("Original array 2D (int)", render_array(a2)),
        card("Transpose array 2D (int)", render_array(a2_t)),
        card("Index Based Transpose array 2D (int)", render_array(a2_t_idx)),
        card("Original array 3D (int)", render_array(a3)),
        card("Transpose array 3D (int)", render_array(a3_t)),
        card("Index 0,2,1 based Transpose array 3D (int)", render_array(a3_t_021)),
        card("Index 1,0,2 based Transpose array 3D (int)", render_array(a3_t_102)),
        card("Index 1,2,0 based Transpose array 3D (int)", render_array(a3_t_120)),
        card("Original array 4D (int)", render_array(a4)),
        card("Transpose array 4D (int)", render_array(a4_t)),
        card("Index 0,1,3,2 based Transpose array 4D (int)", render_array(a4_t_0132)),
        card("Index 0,2,3,1 based Transpose array 4D (int)", render_array(a4_t_0231)),
        card("Index 0,3,1,2 based Transpose array 4D (int)", render_array(a4_t_0312)),
        card("Index 1,0,2,3 based Transpose array 4D (int)", render_array(a4_t_1023)),
    ])
)

# html_doc is the string you already have
output_path = save_html_report(
    __file__,
    "numpy_transpose_report.html",
    html_doc,
    subfolder="reports",
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
