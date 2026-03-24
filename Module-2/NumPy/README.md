
# NumPy — Module 2

This module contains hands‑on exercises and helper utilities to learn **NumPy fundamentals**—array creation & attributes, indexing/slicing, axis manipulation (transpose), and rendering results as **HTML**.

> NumPy is the fundamental package for scientific computing in Python. If you’re new, start with the official quickstart and user guide, then return here to practice.  
> Sources: [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html), [NumPy Tutorials repo](https://github.com/numpy/numpy-tutorials)


---

## Execution

```bash
py -m Module-2.NumPy.numpy_basics_report
py -m Module-2.NumPy.numpy_mathematical_report
py -m Module-2.NumPy.numpy_indexing_slicing
py -m Module-2.NumPy.numpy_transpose_report
```

---

## 🔹 Array Basics & Attributes (0D–4D)

Below are minimal examples that mirror the ndarray model—`ndim`, `shape`, `size`, `dtype`, `itemsize`. *(See NumPy docs for definitions and more examples.)*  
Docs: [Quickstart → The basics](https://numpy.org/doc/stable/user/quickstart.html#the-basics)

```python
import numpy as np

# 0D (scalar)
a0 = np.array(42)
print(a0, a0.ndim, a0.shape)   # 42, 0, ()

# 1D (vector)
a1 = np.array([1, 2, 3, 4])
print(a1, a1.ndim, a1.shape)   # [1 2 3 4], 1, (4,)

# 2D (matrix)
a2 = np.array([[1, 2, 3],
               [4, 5, 6]])
print(a2, a2.ndim, a2.shape)   # 2, (2, 3)

# 3D (stack of matrices)
a3 = np.arange(2*2*3).reshape(2, 2, 3)
print(a3.shape)                # (2, 2, 3)

# 4D (batch of 3D tensors)
a4 = np.arange(2*3*2*2).reshape(2, 3, 2, 2)
print(a4.shape)                # (2, 3, 2, 2)
```

> `ndim`, `shape`, `size`, `dtype`, `itemsize` are the core ndarray attributes you’ll use most frequently.  
> Source: NumPy Quickstart (attributes overview). 

---

## 🔹 Indexing & Slicing — Basic Examples

Indexing and slicing work along each axis. Here are examples including **negative indexing** and **slice ranges**.  
Docs: [Quickstart → Indexing, slicing and iterating](https://numpy.org/doc/stable/user/quickstart.html#basic-operations)

```python
import numpy as np

array_1d = np.array([1, 2, 3, 4, 5, 6])
print(array_1d[3])      # 4 (value at index 3)
print(array_1d[-4])     # 3 (4th from end)
print(array_1d[1:5])    # [2 3 4 5]

array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])
print(array_2d[0, 2])   # 3 (third element first row)
print(array_2d[1, 1])   # 5 (second element second row)
print(array_2d[1, -1])  # 6 (last element second row)

array_3d = np.array([[[1,2,3],[4,5,6]],
                     [[7,8,9],[10,11,12]]])
print(array_3d[1, 0, 0])  # 7 (first elem, first row, second 2D slice)
print(array_3d[1, 1, -1]) # 12 (last elem, last row, last 2D slice)
```

---

## 🔹 Transpose / Axis Permutations — Basic Examples

- **2D**: `a.T` swaps rows and columns: shape `(m, n) → (n, m)`.
- **N‑D**: `np.transpose(a, axes=...)` gives full control over axis order.  
Docs: [ndarray transpose & moveaxis](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)

```python
import numpy as np

# 2D transpose
A = np.array([[1,2,3], [4,5,6]])
print(A.T.shape)  # (3, 2)

# 3D transpose: (D,H,W) -> (H,W,D)
a3 = np.arange(2*2*3).reshape(2,2,3)
B = np.transpose(a3, (1, 2, 0))
print(a3.shape, '->', B.shape)  # (2, 2, 3) -> (2, 3, 2)

# 4D (N,C,H,W) -> (N,H,W,C)
a4 = np.arange(2*3*2*2).reshape(2,3,2,2)
NHWC = np.transpose(a4, (0, 2, 3, 1))
print(a4.shape, '->', NHWC.shape)  # (2,3,2,2)->(2,2,2,3)
```

---

## 🔹 HTML Rendering Helper — `arrays_html.py`

This helper renders arrays and results to a **styled HTML** page or fragment.

### 1) Column‑wise cards for arrays

```python
import numpy as np
from arrays_html import arrays_report_html

arr1 = np.array([[1,2,3,4],[5,6,7,8]])
arr2 = np.array([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.]])
arr3 = np.array([['1','2','3','4'],['5','6','7','8']], dtype=str)

doc = arrays_report_html([
    ("Original array (int)", arr1),
    ("Original array (float)", arr2),
    ("Original array (str)", arr3),
], page_title="Array Details (Column-wise)")

with open("arrays_report.html", "w", encoding="utf-8") as f:
    f.write(doc)
```

### 2) (label, array) pairs — statements + values as HTML

```python
import numpy as np
from arrays_html import pairs_table_html, pairs_report_html

pairs = [
  ("value at index 3 of the 1D NumPy array", np.array([4])),
  ("Employee rating (1:7)", np.array([4,3,5,6,8,9])),
]

# Fragment (embed in notebook or an existing page)
fragment = pairs_table_html(pairs)

# Full page
page = pairs_report_html(pairs, page_title="Indexing & Slicing — Results")
open("pairs_results.html","w",encoding="utf-8").write(page)
```

---

## 🧭 Tips

- Prefer **vectorized operations** over Python loops for speed & clarity.  
- Keep an eye on **shapes** when transposing or stacking—mismatch is a common source of bugs.  
- For reporting, HTML helpers are great to snapshot results for reviews or PRs.

---

## 📚 References

- NumPy user guide & quickstart (arrays, attributes, indexing, axis order):  
  https://numpy.org/doc/stable/user/quickstart.html  
  https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
- NumPy tutorials & notebooks (official project repo):  
  https://github.com/numpy/numpy-tutorials
