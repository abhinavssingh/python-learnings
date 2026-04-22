# NumPy Indexing and Slicing Report

This script demonstrates **indexing, slicing, and negative indexing techniques in NumPy arrays** across **1D, 2D, and 3D arrays**.  
It generates a **visual HTML report** that explains how NumPy accesses data at different dimensions.

The script is designed as a **learning aid** to build strong intuition around NumPy array manipulation.

---

## 📌 Purpose of the Script

The main objectives of this script are to:

- Demonstrate indexing in **1D, 2D, and 3D NumPy arrays**
- Explain **positive and negative indexing**
- Show **slicing patterns** with step values
- Visualize array operations in an **HTML report**
- Reinforce NumPy fundamentals for data science and ML preparation

---

## 🧱 Concepts Covered

---

### 1. One‑Dimensional (1D) Array Indexing

Operations demonstrated:
- Accessing elements using **positive indexing**
- Accessing elements using **negative indexing**
- Performing arithmetic operations on indexed values
- Slicing with:
  - Start, stop, and step
  - Negative ranges
  - Reverse traversal patterns

---

### 2. Two‑Dimensional (2D) Array Indexing

Demonstrates:
- Row‑column indexing using `[row, column]`
- Accessing values using negative indexes
- Row‑wise slicing
- Extracting subsets of rows and columns

---

### 3. Three‑Dimensional (3D) Array Indexing

Demonstrates advanced concepts such as:
- Accessing elements across layers
- Multi‑axis indexing `[layer, row, column]`
- Negative indexing at different dimensions
- Complex slicing across multiple axes

---

## 📊 HTML Report Generation

The script uses a **custom HTML builder** to:

- Render NumPy arrays visually
- Display values clearly using cards
- Organize content in a grid layout
- Improve readability for beginners

This makes NumPy array behavior **easy to understand and verify**.

---

## 📂 Output Details

**Generated HTML File:**
`reports/numpy_indexing_slicing_report.html`
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