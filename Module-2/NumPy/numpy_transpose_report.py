import numpy as np
from arrays_html import arrays_report_html
import webbrowser, os

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
a2_t_idx = a2.transpose(1,0)

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
a3_t_021 = a3.transpose(0,2,1)
a3_t_120 = a3.transpose(1,2,0)
a3_t_102 = a3.transpose(1,0,2)

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
a4_t_0231 = a4.transpose(0,2,3,1)
a4_t_0312 = a4.transpose(0,3,1,2)
a4_t_0132 = a4.transpose(0,1,3,2)
a4_t_1023 = a4.transpose(1,0,2,3)

# Build full column-wise page
html_doc = arrays_report_html([
    ("Original array 0D (int)",   a0),
    ("Original array 1D (int)", a1),
    ("Original array 2D (int)",   a2),
    ("Transpose array 2D (int)",   a2_t),
    ("Index Based Transpose array 2D (int)",   a2_t_idx),
    ("Original array 3D (int)",   a3),
    ("Transpose array 3D (int)",   a3_t),
    ("Index 0,2,1 based Transpose array 3D (int)",   a3_t_021),
    ("Index 1,0,2 based Transpose array 3D (int)",   a3_t_102),
    ("Index 1,2,0 based Transpose array 3D (int)",   a3_t_120),
    ("Original array 4D (int)",   a4),
    ("Transpose array 4D (int)",   a4_t),
    ("Index 0,1,3,2 based Transpose array 3D (int)",   a4_t_0132),
    ("Index 0,2,3,1 based Transpose array 3D (int)",   a4_t_0231),
    ("Index 0,3,1,2 based Transpose array 3D (int)",   a4_t_0312),
    ("Index 1,0,2,3 based Transpose array 3D (int)",   a4_t_1023),
], page_title="Array Details (Column-wise)")

# Save
with open("arrays_transpose_report.html", "w", encoding="utf-8") as f:
    f.write(html_doc)

print("Wrote arrays_transpose_report.html")


webbrowser.open('file://' + os.path.abspath('arrays_transpose_report.html'))