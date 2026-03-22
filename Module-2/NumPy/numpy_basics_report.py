import numpy as np
from arrays_html import arrays_report_html
import webbrowser, os

# Your arrays
arr1 = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8]])

arr1_reshape = arr1.reshape(4,2)
arr1_t = arr1.transpose()

arr2 = np.array([[ 1.,  2.,  3.,  4.],
                 [ 5.,  6.,  7.,  8.],
                 [ 9., 10., 11., 12.]])
arr2_reshape = arr2.reshape(2,3,2)
arr2_t = arr2.transpose()

arr3 = np.array([['1','2','3','4'],
                 ['5','6','7','8']], dtype=str)
arr3_reshape = arr3.reshape(4,2)
arr3_t = arr3.transpose()

# Build full column-wise page
html_doc = arrays_report_html([
    ("Original array (int)",   arr1),
    ("Reshape array (int)",   arr1_reshape),
    ("Transpose array (int)",   arr1_t),
    ("Original array (float)", arr2),
    ("Reshape array (float)",   arr2_reshape),
    ("Transpose array (float)",   arr2_t),
    ("Original array (str)",   arr3),
    ("Reshape array (str)",   arr3_reshape),
    ("Transpose array (str)",   arr3_t),
], page_title="Array Details (Column-wise)")

# Save
with open("arrays_basics_report.html", "w", encoding="utf-8") as f:
    f.write(html_doc)

print("Wrote arrays_basics_report.html")


webbrowser.open('file://' + os.path.abspath('arrays_basics_report.html'))