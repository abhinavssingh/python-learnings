# Capstone-1 – Data Exploration & Optimization Report

This script is a **Capstone-level exercise** that demonstrates end‑to‑end data handling using Pandas, NumPy, and a custom utility layer.  
It focuses on **dataset loading, memory optimization, data transformation, inspection, and reporting**, culminating in an **interactive HTML report**.

---

## 📌 Purpose of This Capstone

The primary objectives of this capstone exercise are to:

- Load a real‑world dataset in a standardized way
- Optimize numeric data types for analytics
- Perform controlled data transformations
- Separate numeric data by type (integer vs float)
- Inspect and document dataset structure
- Generate a **professional, shareable HTML report**

This exercise brings together concepts from:
- NumPy
- Pandas
- DataFrame utilities
- Reporting and presentation

---

## 📂 Dataset Used

**File**
`NSMES1988.csv`

The dataset is loaded using a reusable `DataLoader` utility with:

- Automatic cleanup of unnamed columns
- Memory optimization
- Load‑time metadata reporting

A load report is generated alongside the DataFrame to document:
- Column changes
- Data types
- Memory savings

---

## 🔄 Data Processing Steps

### 1. Dataset Loading & Optimization
- Dataset is loaded with optimization enabled
- A preprocessing report is captured for transparency

### 2. Controlled Data Transformation
A copy of the original DataFrame is created for safe transformations:

- `age` scaled and converted to `float32`
- `income` scaled and converted to `float32`

Why this matters:
- `float16` is efficient for storage and ML tensors
- `float32` is safer for analytics, indexing, and binning

---

### 3. Column Renaming & Formatting
- Columns are renamed for clarity:
  - Age and income are converted to descriptive labels
- All column names are converted to **Title Case**
- Improves readability for reports and stakeholders

---

### 4. Numeric Data Segmentation
Numeric columns are separated into:

- **Unsigned integers** (`uint8`, `uint16`)
- **Floating‑point numbers** (`float16`, `float32`)

This helps with:
- Understanding schema design
- Planning analytical and ML pipelines
- Debugging type‑related issues

---

## 📊 HTML Report Content

The generated HTML report includes:

### ✅ Interactive Dataset Preview
- Collapsible table view for large datasets
- Initial row preview for performance

### ✅ Numeric Subsets
- Integer‑only DataFrame
- Float‑only DataFrame
- Modified DataFrame (after transformations)

### ✅ Dataset Metadata
- `DataFrame.info()` rendered as readable text
- Statistical summary of the modified DataFrame
- Optimization and load report

---

## 🖥 Report Output

The script generates a self‑contained HTML report.

**Output file**
`reports/capstone_1_exercise_report.html`
---
## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `DataLoader` | Dataset loading & optimization |
| `DataFrameHelper` | DataFrame inspection helpers |
| `HtmlBuilder` | Structured HTML output |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Data handling
- Analysis
- Presentation

---