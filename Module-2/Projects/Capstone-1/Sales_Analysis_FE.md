# Sales Analysis Report (Australia – Q4 2020)

This project generates a **comprehensive, interactive sales analysis report** for Australian apparel sales data using **Pandas**, **NumPy**, **Plotly**, and a custom **HTML reporting framework**. The output is a fully interactive HTML dashboard covering data quality, transformations, fiscal analysis, outlier handling, and multi-dimensional visualizations.

---

## 📌 Project Objective

The goal of this analysis is to:

- Clean and enrich raw sales data
- Add **calendar and fiscal attributes** (Australia-specific)
- Detect and handle outliers
- Categorize sales and units into business-friendly bins
- Perform **multi-level exploratory analysis**
- Generate an **interactive HTML report** combining tables, charts, and KPI insights

---

## 📂 Input Dataset

**File:** `AusApparalSales4thQrt2020.csv`

Expected key columns:

- `Date`
- `State`
- `Sales`
- `Unit`
- `Group`
- `Time`

---

## 🛠️ Tech Stack & Libraries

- **Python 3.13+**
- **pandas** – data manipulation
- **numpy** – numerical operations
- **scipy (winsorize)** – outlier treatment
- **plotly (express & graph_objects)** – interactive visualizations
- **Custom utilities**:
  - `DataLoader`
  - `DataFrameHelper`
  - `HtmlBuilder`
  - `PlotRenderer`
  - `ReportUtils`

---

## 🔄 Data Processing Pipeline

### 1. Data Loading

```python
df, report = dl.read_dataset(
    "AusApparalSales4thQrt2020.csv",
    optimize=True,
    handle_unnamed="drop",
    return_report=True
)
```

- Optimized memory usage
- Removed unnamed columns
- Generated input data quality report

---

### 2. Date & Fiscal Enrichment

Using `DataFrameHelper.add_fiscal_calendar` with **Australia** as country:

#### Added Calendar Fields

- Year
- Month name
- Day name
- Weekday
- Is weekend
- Week of month

#### Added Fiscal Fields (Australia FY starts in July)

- Fiscal year
- Fiscal quarter
- Fiscal quarter month range (`FQ_Month_Range`, e.g. `Oct–Dec`)

---

## 3. Feature Engineering

### Sales & Unit Binning

| Metric | Categories           |
| ------ | -------------------- |
| Units  | Very Low → Very High |
| Sales  | Very Low → Very High |

Implemented using `pd.cut`.

---

### Log Transformations

To handle skewness:

- `Log_Unit = log10(Unit)`
- `Log_Sales = log10(Sales)`

---

## 4. Outlier Handling

### Methods Used

- **IQR-based detection** (state-wise)
- **Winsorization** (5% tails, state-wise)

Generated datasets:

- Raw dataframe
- Winsorized dataframe
- Cleaned dataframe (outliers removed)

---

## 📊 Visual Analysis (Plotly)

### Box Plots

- Units and Sales
- With vs without outliers
- State-wise comparison

### Pie Charts

- State-wise sales contribution
- Monthly sales distribution

### Sunburst Chart

Hierarchy:

```
State → Month → Week of Month → Day
```

Displays **percentage contribution of sales** at each level.

---

### Histograms

- Monthly sales (raw vs cleaned)
- Fiscal quarterly sales
- Daily sales
- Time-of-day sales

---

### Heatmaps (% Contribution)

- State × Customer Group
- State × Month
- Time × Customer Group
- Month × Time
- Weekday × State
- Week of Month × State

---

## 📄 HTML Report Generation

The final report includes:

- Collapsible data tables
- Interactive charts
- KPI summaries
- Data quality reports

```python
ru.save_html_report(
    __file__,
    "sales_analysis_report_fe.html",
    html_doc,
    subfolder="reports",
    open_in_browser=True
)
```

---

## 📁 Output

- **HTML Report:** `reports/sales_analysis_report_fe.html`
- Opens automatically in the browser
- Fully interactive (no server required)

---

## ✅ Key Outcomes

- Accurate Australia-specific fiscal reporting
- Clear identification of outliers and skew
- Multi-dimensional view of sales drivers
- Presentation-ready interactive dashboard

---

## 🚀 How to Run

```bash
python sales_analysis_fe.py
```

Ensure all custom `lib/` utilities are available in the project path.

---
