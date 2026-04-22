# Pandas Series Fundamentals Report

This script demonstrates the **fundamentals of Pandas `Series`** and generates a **structured HTML report** summarizing key operations, properties, and selection techniques using Pandas.

It combines **NumPy**, **Pandas**, and a **custom HTML reporting framework** to present results in a readable, browser‑friendly format.

---

## 📌 Purpose of the Script

The main goal of this script is to:

- Teach the basics of **Pandas Series**
- Demonstrate different ways to create a Series
- Showcase indexing, slicing, and filtering techniques
- Apply string operations on Series
- Generate an **HTML report** capturing all observations

---

## 🧱 Key Concepts Covered

### 1. Creating Pandas Series
- From NumPy arrays
- From Python lists (with predefined indexes)
- From dictionaries

### 2. Series Properties
- Data type (`dtype`)
- Shape
- Unique values and counts
- Descriptive statistics (`describe()`)

### 3. Data Access & Selection
- Positional indexing using `iloc`
- Label‑based indexing using `loc`

### 4. String Operations on Series
- Filtering values using:
  - `str.startswith()`
  - `str.contains()`

---

## 📊 HTML Report Generation

The script uses a custom **HTML builder** to create a visual report with:

- Cards for each concept
- Structured grids for layout
- Rendered Series and dictionaries

The report is:
- Saved automatically
- Organized under a `reports/` subfolder
- Opened in the browser after generation

---

## 📂 Output

**Generated File:**
 `reports/pandas_series_report.html`
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