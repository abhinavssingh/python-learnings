# NumPy Transpose Operations – HTML Report

This script demonstrates how **NumPy transpose operations work across different array dimensions (0D to 4D)** and how axis reordering affects array structure.  
The results are presented in a **clear, card‑based HTML report** for visual inspection and learning.

This example is especially useful for understanding **multi‑dimensional data manipulation**, which is critical in data science, image processing, and deep learning.

---

## 📌 Purpose of This Script

The primary goals of this script are to:

- Explain how `transpose()` behaves for arrays of different dimensions
- Demonstrate **axis‑based transposition**
- Compare default transpose vs custom axis permutations
- Show how array rank affects transpose results
- Generate a **visual HTML report** to validate transformations

---

## 🧱 Arrays Covered

### ✅ 0D Array (Scalar)
- A single numeric value
- Demonstrates that transpose has **no effect** on scalars

---

### ✅ 1D Array (Vector)
- A simple 1‑dimensional array
- Shows that transpose does **not change shape** for 1D data

---

### ✅ 2D Array (Matrix)
Shape: `(2, 3)`

Operations demonstrated:
- Standard transpose (`.T` / `.transpose()`)
- Index‑based transpose (`transpose(1, 0)`)

This illustrates **row ↔ column swapping**.

---

### ✅ 3D Array
Shape: `(2, 2, 3)`

Transpose examples include:
- Default transpose (reverses axes)
- Custom axis reordering:
  - `(0, 2, 1)`
  - `(1, 0, 2)`
  - `(1, 2, 0)`

These examples explain how data moves across **depth, rows, and columns**.

---

### ✅ 4D Array
Shape: `(N, C, H, W)` style structure  
Used commonly in:
- Image data
- Deep learning tensors

Custom transposes demonstrated:
- Channel‑last vs channel‑first
- Spatial axis swapping
- Batch dimension rearrangement

Examples:
- `transpose(0, 2, 3, 1)`
- `transpose(0, 3, 1, 2)`
- `transpose(0, 1, 3, 2)`
- `transpose(1, 0, 2, 3)`

---

## 🔄 Transpose Concepts Demonstrated

- Default transpose reverses axes
- Axis order defines **how dimensions are rearranged**
- Higher‑dimensional arrays provide flexibility but require care
- Incorrect axis order can lead to:
  - Wrong model inputs
  - Incorrect mathematical operations

---

## 🖥 HTML Report Output

The script generates a **detailed HTML report** showing:

- Original arrays at each dimension
- Corresponding transposed versions
- Explicit labeling of axis‑based transposes
- Easy, visual comparison of shapes and values

### Output File
**Generated HTML File:**
`reports/numpy_transpose_report.html`
---
## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `HtmlBuilder` | Structured HTML output |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Analysis
- Presentation

---