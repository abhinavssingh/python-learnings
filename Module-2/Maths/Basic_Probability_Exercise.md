# Probability Exercise Report – Retail Store Data Analysis

This script performs a **practical probability exercise using real retail store data** to demonstrate how **probability distributions are applied to real-world business scenarios**.  
It generates a **visual, interactive HTML report** illustrating discrete and continuous probability models using Plotly.

The focus is on **connecting probability theory with data-driven decision-making**.

---

## 📌 Purpose of the Script

The main objectives of this exercise are to:

- Apply probability distributions to a **retail dataset**
- Convert raw business data into **probabilistic events**
- Model customer behavior using:
  - Bernoulli
  - Binomial
  - Poisson
  - Normal
  - Uniform distributions
- Compare **empirical data vs theoretical distributions**
- Generate an **interactive HTML probability report**

---

## 🧱 Dataset Overview

The script uses:

**Dataset:** `Retail_Store_Data.csv`

Typical columns include metrics such as:
- `Visit_Duration`
- `Purchase_Amount`
- Other transactional attributes

A custom `DataLoader` utility:
- Optimizes memory usage
- Drops unnamed columns
- Produces a dataset loading report

---

## 📊 Probability Distributions Covered

### 1. Bernoulli Distribution
**Question answered:**  
Did the customer spend **more than 30 minutes** in the store?

- Binary outcome:  
  - `0` → ≤ 30 minutes  
  - `1` → > 30 minutes
- Histogram shows probability mass
- KDE curve added for smooth visualization

Used to model **single success / failure events**.

---

### 2. Binomial Distribution
**Question answered:**  
Out of **10 customer visits**, how many result in **high spending (> $100)**?

- Probability `p` estimated from full dataset
- Binomial PMF plotted
- KDE added using simulated samples

Used to model **multiple Bernoulli trials**.

---

### 3. Poisson Distribution
**Question answered:**  
How many customers visit the store per hour?

- Assumes average rate λ = 15 customers/hour
- Poisson PMF plotted
- KDE added using simulated observations

Used to model **event counts over time**.

---

### 4. Normal Distribution
**Question answered:**  
How is the purchase amount distributed?

- Mean (μ) and standard deviation (σ) calculated from data
- Normal PDF plotted
- Histogram and empirical KDE overlaid

Used to model **continuous data around a mean**.

---

### 5. Uniform Distribution
**Question answered:**  
Is visit duration uniformly spread between min and max?

- Uniform PDF created from data range
- Histogram and KDE compared against theory

Used to model **equal-probability intervals**.

---

## 📈 Visualization Strategy

Each distribution includes:

- Histogram or bar chart
- Theoretical PMF / PDF curve
- KDE (Kernel Density Estimation) overlay

This allows comparison between:
- Mathematical theory
- Actual observed data
- Smoothed empirical behavior

All charts are fully interactive (zoom, hover, legend toggle).

---

## 🖥 HTML Report Output

The script generates a **single interactive HTML report** containing:

- Dataset preview (collapsible)
- Statistical summary (`describe`)
- DataFrame metadata (`info`)
- All probability distribution charts

## Output file
`reports/basic_probability_exercise_report.html`
---
## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `DataLoader` | Dataset loading & optimization |
| `DataFrameHelper` | DataFrame inspection helpers |
| `HtmlBuilder` | Structured HTML output |
| `PlotRenderer` | Structured Plot output |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Data handling
- Analysis
- Presentation

---
 