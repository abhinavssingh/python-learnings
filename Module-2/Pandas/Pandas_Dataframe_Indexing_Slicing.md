# Pandas DataFrame Indexing and Slicing Report

This script demonstrates **indexing, slicing, and conditional selection techniques in Pandas DataFrames** and automatically generates a **visual HTML report** summarizing the results.

It is designed as a hands‑on learning artifact to understand how Pandas accesses data using labels, positions, and conditions.

---

## 📌 Purpose of the Script

The primary objective of this script is to:

- Explain **row and column selection** in Pandas DataFrames
- Compare `loc`, `iloc`, `at`, and `iat`
- Demonstrate **conditional filtering**
- Present results in a **structured HTML report**

---

## 🧱 Concepts Covered

### 1. DataFrame Creation
- Creates a `5 × 5` DataFrame using NumPy random integers
- Uses **custom row and column labels** for clarity

### 2. Column and Row Selection
- Selects a single column using `df['Col_A']`
- Selects a single row using `df.loc['Row_1']`
- Selects multiple columns and row ranges using label‑based slicing

---

### 3. Indexing Methods Explained

| Method | Purpose |
|------|-------|
| `loc` | Label‑based row and column access |
| `iloc` | Position‑based indexing |
| `at` | Fast access to a single scalar value (label‑based) |
| `iat` | Fast access to a single scalar value (position‑based) |

---

### 4. Conditional Selection
This script illustrates filtering based on conditions such as:
- Rows where **Column A > 10**
- Values in a specific row meeting a threshold
- Rows meeting **both column and index conditions**

---

## 📊 HTML Report Generation

A custom **HTML rendering framework** is used to:

- Display DataFrames and Series visually
- Organize outputs into cards and grids
- Make learning outputs browser‑friendly

---

## 📂 Output Details

**Generated HTML File:**
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