# Pandas Date Range & Fiscal Calendar Report

This script demonstrates how to generate a **date-driven Pandas DataFrame**, enrich it with **country‑specific fiscal calendar attributes**, perform **time‑based aggregations**, optimize data types, and publish the results as a **structured HTML report**.

It is designed as an **advanced Pandas learning example**, especially useful for analytics, reporting, and financial calendar use cases.

---

## 📌 Purpose of This Script

The primary goals of this script are to:

- Generate a continuous date range using Pandas
- Enrich dates with **fiscal calendar fields (India-specific)**
- Perform calendar and fiscal aggregations
- Analyze weekends vs weekdays
- Optimize DataFrame memory usage
- Produce a **professional HTML analytics report**

---

## 🧱 Key Features & Concepts

### 1. Date Range DataFrame Creation

A daily date range is created using:

```python
pd.date_range(start="2024-01-01", end="2026-12-31", freq="D")
```
---
## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `DataFrameHelper` | DataFrame inspection helpers |
| `HtmlBuilder` | Structured HTML output |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Data handling
- Analysis
- Presentation

---