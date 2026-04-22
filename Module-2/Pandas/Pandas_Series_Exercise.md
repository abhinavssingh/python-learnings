# Pandas Series Exercise – Sales Analysis Report

This script demonstrates **practical exercises using Pandas Series** with a simple **weekly sales dataset**, applying descriptive statistics, filtering techniques, ranking, trend analysis, and **outlier detection methods**.  
The results are presented in a **structured HTML report** for clear interpretation and learning.

---

## 📌 Purpose of This Script

The primary objectives of this script are to:

- Practice real‑world operations on a Pandas `Series`
- Perform descriptive statistics on indexed data
- Demonstrate label‑based and pattern‑based selection
- Identify outliers using multiple statistical approaches
- Analyze day‑over‑day changes
- Generate a professional **HTML report** summarizing insights

---

## 🧱 Dataset Description

The script uses a **weekly sales dataset**:

- **Values:** Daily sales numbers
- **Index:** Days of the week (Monday–Sunday)
- Sunday sales are intentionally updated to create a noticeable outlier

This setup simulates a **business sales scenario** commonly used in analytics exercises.

---

## 🔍 Series Operations Demonstrated

### 1. Series Creation & Indexing
- Creation of a Pandas Series with custom string labels
- Updating values using index labels
- Selecting specific days using `.loc[]`

---

### 2. Label‑Based & Pattern‑Based Filtering
- Selecting preferred days (e.g. Monday, Wednesday, Friday)
- Filtering index labels using string matching (`contains`)
- Combining multiple label conditions

---

### 3. Descriptive Statistics
The following metrics are calculated:
- Total sales
- Average (mean) sales
- Maximum and minimum sales
- Corresponding day labels
- Standard deviation
- Quartiles (Q1, Q3) and IQR

These provide a complete **summary of central tendency and spread**.

---

## 🚨 Outlier Detection Techniques

The script demonstrates **three different approaches** to detect outliers:

### ✅ IQR (Interquartile Range) Method
- Calculates lower and upper fences
- Identifies values far from the middle 50%

### ✅ Z‑Score Method
- Standardizes values using mean and standard deviation
- Flags values beyond ±2 sigma
- Maps Z‑score outliers back to actual sales values

### ✅ Robust Z‑Score (MAD Method)
- Uses median and Median Absolute Deviation (MAD)
- More resilient to extreme values

These methods highlight how **different statistical techniques may detect outliers differently**.

---

## 📈 Trend & Ranking Analysis

Additional analytical features include:

- **Day‑over‑day percentage change**
- **Absolute change between days**
- **Descending rank of sales values**

This helps identify:
- Highest performing days
- Sudden spikes or dips
- Relative performance ordering

---

## 🖥 HTML Report Output

The script generates a detailed HTML report containing:

- Original sales series
- Filtered and selected views
- Summary metrics
- Outlier detection results
- Z‑scores and robust Z‑scores
- Recommendations for handling outliers

### Output File
**Generated File:**
 `reports/pandas_series_exercise_report.html`
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