# Pandas Series Maths Operations – HTML Report

This script demonstrates **mathematical and logical operations on Pandas Series** and generates a **well-structured HTML report** summarizing the results.

It is designed as a **learning-focused example** to understand how Pandas `Series` behave during arithmetic operations, filtering, sorting, and missing value handling.

---

## 📌 Purpose of This Script

The primary objectives of this script are to:

- Perform **element-wise mathematical operations** on Pandas Series
- Handle missing values (`NaN`) correctly
- Apply transformations and conditions on Series data
- Demonstrate conditional filtering techniques
- Generate a structured **HTML report** for visualization and learning

---

## 🧱 Key Concepts Demonstrated

### 1. Series Creation
Two Pandas Series are created using Python lists and NumPy:

- `s1` includes numeric values **with a NaN**
- `s2` contains numeric values only

This helps illustrate how Pandas handles:
- Alignment by index
- Missing values during operations

---

### 2. Mathematical Operations on Series

The following element-wise operations are performed:

- ✅ Addition (`s1 + s2`)
- ✅ Subtraction (`s2 - s1`)
- ✅ Multiplication (`s1 * s2`)
- ✅ Division (`s2 / s1`) with rounding applied

These operations demonstrate **automatic index alignment** and `NaN` propagation.

---

### 3. Applying Functions

A lambda function is applied to every element of a series:

```python
squared_series = s1.apply(lambda x: x**2)
```
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