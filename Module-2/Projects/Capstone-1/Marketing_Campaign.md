# Marketing Campaign Report – Data Overview & Optimization

This script generates an **interactive HTML report** for a marketing campaign dataset.  
It focuses on **data loading, optimization, inspection, and presentation**, providing a clean, readable summary suitable for analytics review and reporting.

The script is intentionally lightweight and serves as a **baseline report** for understanding dataset structure before deeper analysis or visualization.

---

## 📌 Purpose of This Script

The main objectives of this report are to:

- Load a marketing campaign dataset in a standardized manner
- Optimize dataset memory usage
- Inspect dataset structure and schema
- Provide an interactive preview for large datasets
- Produce a clean, shareable HTML report

This script is typically used as:
- A **first‑pass exploratory report**
- A **data quality and structure check**
- A starting point for marketing analytics workflows

---

## 📂 Dataset Used

**File**
`marketing_data.csv`

The dataset is loaded using a reusable `DataLoader` utility with:

- Automatic removal of unnamed columns
- Memory optimization
- A structured optimization and load report

---

## 🔄 Processing Steps

### 1️⃣ Dataset Loading
- Dataset is loaded from CSV
- Unnamed columns are removed
- Data types are optimized to reduce memory footprint

### 2️⃣ Dataset Inspection
- `DataFrame.info()` is captured as a readable string
- Provides details on:
  - Column names
  - Data types
  - Non‑null counts
  - Memory usage

### 3️⃣ Interactive Data Preview
- The full dataset is rendered using a **collapsible table**
- Initial rows are shown for performance
- Suitable for large marketing datasets

---

## 🖥 HTML Report Content

The generated report includes:

- ✅ Interactive, collapsible dataset preview
- ✅ DataFrame structure and metadata
- ✅ Optimization and load report

This design ensures the report is:
- Easy to scan
- Suitable for stakeholders
- Safe for large datasets

---

## 📄 Report Output

The script generates a self‑contained HTML report.

**Output file**
`reports/marketing_campaign_report.html`
---

## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `DataLoader` | Dataset loading & optimization |
| `DataFrameHelper` | DataFrame inspection helpers |
| `HtmlBuilder` | Structured HTML layout |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure a clean separation between:
- Data handling
- Inspection logic
- Presentation
---