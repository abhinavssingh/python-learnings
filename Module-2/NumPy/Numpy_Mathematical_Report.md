# NumPy Mathematical & Statistical Operations – HTML Report

This script demonstrates **core mathematical and statistical operations using NumPy** on both **2D and 3D arrays**, and presents the results in a **structured, human‑readable HTML report**.

It is designed as a **foundational learning exercise** to understand how NumPy performs element‑wise arithmetic and statistical computations across different array dimensions.

---

## 📌 Purpose of This Script

The primary goals of this script are to:

- Perform **element‑wise mathematical operations** on NumPy arrays
- Work with both **2D and 3D arrays**
- Apply **basic statistical functions**
- Compare numeric behavior across dimensions
- Generate an **HTML report** to visualize calculations clearly

---

## 🧱 Arrays Used

### ✅ 2D Arrays
Two integer matrices of shape `(2, 4)` are used:
- `arr1`
- `arr2`

These are used for:
- Arithmetic operations
- Statistical calculations

---

### ✅ 3D Arrays
Two identical 3D arrays of shape `(2, 2, 3)` are used:
- `a3`
- `array_3d`

These demonstrate:
- Multi-dimensional arithmetic
- Statistics across higher-dimensional data

---

## ➕ Mathematical Operations Demonstrated

All operations are performed **element‑wise**:

- Addition (`np.add`)
- Subtraction (`np.subtract`)
- Multiplication (`np.multiply`)
- Division (`np.divide`)
- Rounded Division (`np.round`)
- Power (`np.power`)

Operations are applied consistently to both **2D and 3D arrays**, highlighting how NumPy automatically broadcasts when shapes are compatible.

---

## 📊 Statistical Operations Demonstrated

For both 2D and 3D arrays, the following statistics are calculated:

- **Mean**
- **Median**
- **Standard Deviation**
- **Variance**
- **80th Percentile**
- **90th Percentile**

These metrics help understand:
- Central tendency
- Data dispersion
- Distribution behavior

---

## 🖥 HTML Report Output

The script compiles all results into a **card‑based HTML report**, including:

- Original arrays
- Results of each mathematical operation
- Statistical summaries for each array
- Clear separation of 2D and 3D array analysis

### Output File
**Generated HTML File:**
`reports/numpy_mathematical_report.html`
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