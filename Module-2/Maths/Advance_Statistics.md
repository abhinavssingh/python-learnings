# Advanced Statistics Operation Report

This script performs a **comprehensive set of hypothesis testing and statistical analyses** using real‑world datasets and generates a **professional, interactive HTML report**.

It combines **classical statistical theory**, **practical Python implementation**, and **distribution‑based visual explanations** to make hypothesis testing both **accurate and interpretable**.

---

## 🎯 Objective

The primary objectives of this module are to:

- Apply the correct hypothesis test for different data scenarios
- Demonstrate statistical decision‑making using p‑values and critical regions
- Visualize test statistics against:
  - F‑distributions
  - Chi‑square distributions
- Generate a single, well‑structured **HTML analytics report**

This script is suitable for:
- Data science learning
- Quality & process analysis
- Statistical reporting
- Interview and portfolio projects

---

## 📂 Datasets Used

The following datasets are analyzed:

- **Cutlets** – Diameter comparison between two production units
- **Ages** – One‑sample mean testing
- **Blood Pressure** – Paired observations (before vs after treatment)
- **Employee Satisfaction** – Two‑sample comparison across departments
- **Plant Weights** – One‑way ANOVA across treatment groups
- **Crop Yield** – Two‑way ANOVA (fertilizer × water)
- **Customer Order Form** – Defective vs error‑free analysis across centers
- **Chi‑Test Dataset** – Association between two categorical variables

All datasets are loaded using a standardized `DataLoader` with optimization enabled.

---

## 🧪 Statistical Tests Implemented

### ✅ One‑Sample T‑Test
- Tests whether a sample mean differs from a known population mean  
- Example: Employee age claim validation

### ✅ Paired T‑Test
- Tests mean difference between paired observations  
- Example: Blood pressure before and after treatment

### ✅ Two‑Sample Independent T‑Test
- Compares means between two independent groups  
- Example: Employee satisfaction (Sales vs Marketing)

### ✅ Z‑Test
- Tests population mean when variance is known or sample size is large  
- Example: IQ score claim testing

### ✅ Chi‑Square Test of Independence
- Tests association between categorical variables  
- Example: Gender vs shopping preference  
- Example: Order defects vs processing center

### ✅ One‑Way ANOVA (F‑Test)
- Compares means across multiple groups  
- Example: Plant weight comparison  
- Example: Cutlet diameter comparison

### ✅ Two‑Way ANOVA
- Examines main effects and interaction effects  
- Example: Crop yield by fertilizer and water level

---

## 📈 Distribution‑Based Visualizations

To clearly support hypothesis decisions, this script generates:

### 🔵 F‑Distribution Plots
Used for:
- One‑Way ANOVA
- Two‑Way ANOVA
- Variance‑based hypothesis tests

Plots include:
- Theoretical F‑distribution curve
- Critical value (α = 0.05)
- Observed F‑statistic
- Multi‑DF comparison charts

### 🔴 Chi‑Square Distribution Plots
Used for:
- Categorical hypothesis testing

Plots include:
- Chi‑square distribution curve
- Critical value
- Observed χ² statistic
- Multiple degrees‑of‑freedom comparison charts

All plots are generated using `DistributionPlotHelper`.

---

## 🧠 Hypothesis Testing Framework

Each analysis follows a consistent structure:

1. Define null hypothesis (H₀) and alternative hypothesis (H₁)
2. Choose significance level (α = 0.05)
3. Select the appropriate statistical test
4. Compute test statistic
5. Calculate p‑value or critical value
6. Make statistical decision
7. Provide an interpretable conclusion

These steps are explicitly documented in the report.

---

## 🖥 HTML Report Output

The script generates a **single consolidated HTML report** containing:

- Mathematical formulas (via `FORMULA_REGISTRY`)
- Input datasets (rendered as tables)
- Hypothesis testing results (as dictionaries)
- F‑distribution plots
- Chi‑square distribution plots
- Statistical explanations and interpretations
---

**Output file**
`reports/advance_statistics_operation_report.html`
---
## 🛠 Libraries & Utilities Used

### Scientific Libraries
- NumPy
- Pandas
- SciPy
- StatsModels

### Project Utilities
| Component | Purpose |
|--------|--------|
| `DataLoader` | Dataset loading & optimization |
| `HtmlBuilder` | Structured HTML report generation |
| `PlotRenderer` | Plotly chart embedding |
| `DistributionPlotHelper` | Statistical distribution plots |
| `FORMULA_REGISTRY` | Formula rendering |
| `ReportUtils` | Save & open reports |
