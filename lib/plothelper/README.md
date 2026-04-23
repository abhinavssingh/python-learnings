# DistributionPlotHelper

`DistributionPlotHelper` is a reusable utility class designed to create **statistical distribution visualizations** using **Plotly**.  
It focuses on **F‑distributions** and **Chi‑Square distributions**, which are commonly used in **ANOVA, F‑tests, and Chi‑Square hypothesis testing**.

This helper is intended for **educational, analytical, and reporting use cases**, where visual interpretation of hypothesis testing is important.

---

## 🎯 Purpose

The main goals of this helper are to:

- Visualize **F‑distribution** curves for ANOVA and F‑tests
- Visualize **Chi‑Square distribution** curves for categorical hypothesis testing
- Clearly mark:
  - Observed test statistic
  - Critical value (based on α)
- Support both **single** and **multiple degrees of freedom** comparisons
- Integrate seamlessly with HTML‑based analytical reports

---

## 📦 Dependencies

This helper relies on:

- `numpy`
- `scipy.stats` (`f`, `chi2`)
- `plotly.graph_objects`

Make sure these are installed:

```bash
pip install numpy scipy plotly
```