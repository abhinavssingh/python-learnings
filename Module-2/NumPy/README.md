
# Module 2 – NumPy

This module contains **hands‑on Python scripts for learning NumPy**, focusing on **array creation, reshaping, transpose operations, mathematical computations, and statistical analysis**.

Each script generates a **self‑documenting HTML report**, making it easier to understand and verify NumPy operations visually.

---

## 📌 Objective of This Module

The goals of this NumPy module are to:

- Build a strong foundation in **NumPy array mechanics**
- Understand how NumPy handles **multi‑dimensional data (0D → 4D)**
- Perform **mathematical and statistical operations** efficiently
- Learn **axis‑based transformations** such as transpose
- Present numerical results through **structured HTML reports**

---

## 📂 Folder Structure

This folder contains multiple Python scripts, each focused on a specific NumPy concept:

### 🧮 1. NumPy Basics
**What it covers:**
- Array creation with different data types (int, float, string)
- `reshape()` operations
- `transpose()` operations
- Understanding shape preservation vs structural change

**Key Learnings:**
- NumPy arrays are type‑agnostic in structure
- Reshape changes dimensions, not values
- Transpose reorders axes

---

### ➕ 2. NumPy Mathematical Operations
**What it covers:**
- Element‑wise arithmetic:
  - Addition
  - Subtraction
  - Multiplication
  - Division
  - Power
- Operations on **2D and 3D arrays**
- Rounding and safe numeric calculations

**Key Learnings:**
- NumPy performs vectorized operations efficiently
- Broadcasting works when shapes are compatible
- Mathematical logic scales naturally to higher dimensions

---

### 📊 3. NumPy Statistical Operations
**What it covers:**
- Mean
- Median
- Standard deviation
- Variance
- Percentiles (80th, 90th)

Applied to:
- 2D arrays
- 3D arrays

**Key Learnings:**
- Statistical functions behave consistently across dimensions
- Understanding data spread and distribution is essential for data science

---

### 🔄 4. NumPy Transpose Operations
**What it covers:**
- Transpose behavior for:
  - 0D arrays (scalars)
  - 1D arrays (vectors)
  - 2D arrays (matrices)
  - 3D and 4D tensors
- Axis‑based transpose using custom index orders

**Key Learnings:**
- Transpose has no effect on 0D and 1D arrays
- Higher‑dimensional transpose is **axis reordering**
- Critical for:
  - Machine learning
  - Image processing
  - Tensor manipulation

---

## 🖥 HTML Reporting Pattern

All scripts follow a **consistent reporting architecture**:

- NumPy computation logic
- Structured HTML output using reusable utilities
- Card‑based visual layout for clarity
- Automatic browser opening after report generation

---

## Execution

```bash
py -m Module-2.NumPy.numpy_basics_report
py -m Module-2.NumPy.numpy_mathematical_report
py -m Module-2.NumPy.numpy_indexing_slicing
py -m Module-2.NumPy.numpy_transpose_report
```
---

## 📚 References

- NumPy user guide & quickstart (arrays, attributes, indexing, axis order):  
  https://numpy.org/doc/stable/user/quickstart.html  
  https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
- NumPy tutorials & notebooks (official project repo):  
  https://github.com/numpy/numpy-tutorials
