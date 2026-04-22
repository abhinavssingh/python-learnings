# Pandas DataFrame Maths & Operations Report

This script demonstrates **mathematical operations, transformations, concatenation, merging, and sorting** on Pandas DataFrames.  
It generates a **comprehensive HTML report** that visually presents the results of each operation for easier learning and comparison.

---

## 📌 Purpose of the Script

The objective of this script is to:

- Perform **element‑wise mathematical operations** on DataFrames
- Apply **NumPy functions and lambda expressions** to DataFrames
- Demonstrate **DataFrame concatenation and merging**
- Explain **different join types** (inner, outer, left, right)
- Present all results in a **structured HTML report**

---

## 🧱 Concepts Covered

### 1. DataFrame Creation
- Two `5 × 5` DataFrames are created using NumPy random integers
- Custom row and column labels are applied for clarity

---

### 2. Arithmetic Operations (Element‑wise)
The following mathematical operations are performed:

- ➕ Addition
- ➖ Subtraction
- ✖ Multiplication
- ➗ Division (with zero‑safe handling using `NaN`)
- 🔢 Power operation (`x²`)

---

### 3. Function Application
- Apply **NumPy functions** (`sqrt`) across DataFrames
- Apply **lambda expressions** to transform values
- Demonstrates column‑wise application using `apply()`

---

### 4. DataFrame Concatenation (`concat`)
The script demonstrates concatenation:
- Along rows (`axis=0`)
- Along columns (`axis=1`)
- With ignored indexes
- Using **hierarchical keys**

---

### 5. DataFrame Merging (`merge`)
A common `Key` column is introduced to demonstrate joins:

| Merge Type | Description |
|-----------|-------------|
| Inner | Intersection of keys |
| Outer | Union of keys |
| Left | Preserves left DataFrame |
| Right | Preserves right DataFrame |

Also demonstrates:
- Key‑based intersection
- Column suffix handling

---

### 6. Sorting
- Sorting DataFrames based on specific column values
- Demonstrates real‑world data ordering scenarios

---

## 📊 HTML Report Generation

A custom **HTML rendering framework** is used to:

- Display DataFrames clearly
- Arrange outputs in grid‑based cards
- Improve readability for learning and evaluation

---

## 📂 Output Details

**Generated File:**
`reports/pandas_dataframe_maths_report.html`
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