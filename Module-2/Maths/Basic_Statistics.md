# Basic Statistics Operations Report

This script demonstrates **core statistical concepts** using NumPy, SciPy, and Plotly, and generates an **interactive HTML report** that combines **mathematical formulas, numerical results, and statistical visualizations**.

It is designed as a **learning‑oriented statistics notebook in script form**, suitable for data science, analytics, and machine learning foundations.

---

## 📌 Purpose of the Script

The primary goals of this script are to:

- Compute **basic descriptive statistics**
- Distinguish between **population vs sample statistics**
- Explain **variance, covariance, and correlation**
- Demonstrate **skewness and kurtosis**
- Visualize **data distributions with KDE**
- Generate a **rich, interactive HTML statistics report**

---

## 🧱 Concepts Covered

---

### 1. Descriptive Statistics (NumPy)

The script computes key measures such as:

- Mean (population & sample)
- Median
- Variance
- Standard deviation
- Covariance
- Correlation

It clearly distinguishes:
- `ddof=0` → population statistics  
- `ddof=1` → sample statistics  

---

### 2. Higher‑Order Statistics (SciPy)

Using `scipy.stats`, the script calculates:

- **Skewness**
  - Measures asymmetry of the distribution
- **Kurtosis**
  - Measures tail‑heaviness of the distribution

Both **population and sample interpretations** are demonstrated.

---

### 3. Synthetic Distribution Analysis

Multiple distributions are generated to explain shape behavior:

| Distribution | Property |
|--------------|---------|
| Normal | Symmetric |
| Exponential | Right‑skewed |
| Negative Exponential | Left‑skewed |
| Student‑t | Heavy‑tailed |

Each distribution is analyzed for:
- Skewness
- Kurtosis
- Mean vs Median behavior

---

### 4. Kernel Density Estimation (KDE)

For each distribution:
- KDE curves are plotted using `gaussian_kde`
- Helps visualize the **true underlying density**
- Compared alongside histograms

---

### 5. Visualization with Plotly

Each subplot includes:
- Histogram (probability density)
- KDE curve
- Mean (dashed line)
- Median (dotted line)
- Text annotation for skewness & kurtosis

This makes statistical properties **visually intuitive**.

---

### 6. Mathematical Formula Registry

The report automatically renders:
- Statistical formulas from a central `FORMULA_REGISTRY`
- Categorized under **Statistics**
- Ensures conceptual + mathematical clarity

---

## 📊 HTML Report Features

The generated HTML report includes:

- Formula cards
- Computation cards
- Interactive Plotly charts
- Grid‑based layout
- Clean mathematical presentation

This makes it ideal for:
- Learning
- Revision
- Teaching
- Documentation

---

## 📂 Output Details

**Generated Report File:**
`reports/basic_statistics_operation_report.html`
---
## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `FORMULA_REGISTRY` | Automatic registration of formulas and rendering |
| `HtmlBuilder` | Structured HTML output |
| `PlotRenderer` | Structured Plot output |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Data handling
- Analysis
- Presentation

---