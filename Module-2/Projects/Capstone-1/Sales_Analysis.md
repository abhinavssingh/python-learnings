# Sales Analysis Report – Interactive Data Visualization

This script performs an **end‑to‑end sales analysis** on an Australian apparel sales dataset and produces a **fully interactive HTML analytics report** using **Pandas, NumPy, and Plotly**.

The report combines **time‑based analysis, fiscal calendars, aggregations, and advanced visualizations**, closely resembling a **BI dashboard built in tools like Power BI or Tableau**, but implemented entirely in Python.

---

## 📌 Purpose of This Analysis

The main objectives of this script are to:

- Load and optimize a real‑world sales dataset
- Enrich the data with **calendar and fiscal attributes**
- Aggregate sales across multiple business dimensions
- Explore relationships using a wide variety of visualizations
- Generate a **shareable HTML report** for insights and decision‑making

---

## 📂 Dataset Used

**File**
`AusApparalSales4thQrt2020.csv`

**Key fields used**
- Date
- State
- Group (Customer Group)
- Sales
- Units
- Time (time of day)
- Visits / visit‑related metrics

The dataset is loaded using a reusable `DataLoader` utility with:
- Removal of unnamed columns
- Numeric type optimization
- Memory optimization report

---

## 🔄 Data Preparation & Feature Engineering

### 1️⃣ Datetime Conversion
- `Date` column is converted to `datetime`
- Enables time‑based grouping and analysis

### 2️⃣ Calendar & Fiscal Enrichment
Fiscal and calendar fields are added using `DataFrameHelper`, including:

- Year
- Month name
- Day name
- Weekday flag
- Weekend flag
- Fiscal year
- Fiscal quarter

This enables:
- Seasonal analysis
- Quarterly reporting
- Weekday vs weekend insights

---

### 3️⃣ Numeric Optimization
- Numeric columns are re‑optimized after feature creation
- Improves memory efficiency and performance

---

## 📊 Aggregations Performed

Multiple grouped datasets are created to support visualization, including:

- Time of day × Customer group
- State × Customer group
- State × Month
- Month × Time of day
- Day of week × State
- Correlation matrix for numeric fields

These aggregations power heatmaps, distributions, and comparisons.

---

## 📈 Visualizations Included

The report uses **Plotly** for **fully interactive charts**, including:

### 🔥 Heatmaps
- Time of day vs Customer group
- State vs Customer group (percentage distribution)
- State vs Month
- Weekday vs State
- Month vs Time of day
- Correlation heatmap (Sales vs Units)

---

### 📊 Histograms
- Monthly sales by state
- Fiscal quarter sales by state
- Faceted by state and sales volume

---

### 🎻 Violin Plots
- Units distribution by state and year
- Sales distribution by state and year  
Used to show spread, density, and outliers

---

### 🧁 Pie Chart
- State‑wise sales contribution  
Shows proportional sales distribution

---

### 🌞 Sunburst & Treemap
- Hierarchical views of sales by:
  - State
  - Month
  - Customer group
- Useful for seeing **relative contribution at multiple levels**

---

### 🔗 Correlation Plot
- Correlation between numeric metrics (Units & Sales)
- Helps identify linear relationships

---

## 🖥 HTML Report Output

The script generates a **single interactive HTML report** containing:

- Collapsible dataset preview
- Dataset metadata and statistical summary
- All charts arranged in a responsive grid
- Hover, zoom, filter, and legend interactivity

**Output file**
`reports/sales_analysis_report.html`
---

## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `DataLoader` | Dataset loading & optimization |
| `DataFrameHelper` | Calendar, fiscal, and DataFrame helpers |
| `HtmlBuilder` | Structured HTML layout |
| `PlotRenderer` | Embed Plotly charts |
| `ReportUtils` | Save & open reports |

---