# Marketing Campaign Analytics & Hypothesis Testing Report

## 📌 Project Overview

This project performs **end-to-end marketing campaign analytics** on customer data using Python.
It includes **data cleaning, feature engineering, statistical hypothesis testing, correlation analysis, and interactive visualizations**, culminating in an **automatically generated HTML report**.

The analysis aims to uncover **customer behavioral patterns, purchasing preferences, campaign effectiveness, and demographic insights** to support data‑driven decision-making.

---

## 🧰 Tech Stack & Libraries

### Core Libraries

- **Python**
- **NumPy**
- **Pandas**
- **Plotly Express**
- **SciPy**
- **Statsmodels**
- **Scikit‑learn**

### Custom Utility Modules

- `HtmlBuilder` – Builds structured HTML reports
- `PlotRenderer` – Renders Plotly charts into HTML cards
- `DataLoader` – Optimized dataset loading with reporting
- `DataFrameHelper` – Feature engineering & outlier handling
- `ReportUtils` – Saves and launches HTML reports

---

## 📂 Dataset Description

Input file:`marketing_data.csv`

### Key Data Categories

#### Demographic Attributes

- Age (derived)
- Age_Category (`Young`, `Mid-Age`, `Senior`, `Elder`)
- Education
- Marital_Status
- Country
- Income
- Income_Category
- Kidhome
- Teenhome

#### Behavioral Metrics

- NumWebPurchases
- NumStorePurchases
- NumCatalogPurchases
- NumDealsPurchases

#### Product-Level Spending

- MntWines
- MntFruits
- MntMeatProducts
- MntFishProducts
- MntSweetProducts
- MntGoldProds

#### Campaign & Feedback

- AcceptedCmp1–AcceptedCmp5
- Complain (complaints in last two years)

---

## ⚙️ Data Cleaning & Feature Engineering

### 1️⃣ Income Processing

- Trimmed whitespace from column names
- Removed `$` and `,`
- Converted Income to numeric
- **Group-wise mean imputation** based on Education & Marital Status

### 2️⃣ Derived Features

- **Log_Income** (log10 transformed)
- **Totalchildren** = Kidhome + Teenhome
- **Age** derived from enrollment date
- **TotalSpend** = Sum of all `Mnt*` columns
- **TotPurchase** = Total purchases across Store, Web, Catalog

### 3️⃣ Categorical Encoding

- **Quantile-based categories** for Age and Income
- Ordinal coding for Age & Income categories
- **One-Hot Encoding** for Education and Marital Status using `OneHotEncoder`

### 4️⃣ Outlier Treatment

- IQR-based outlier detection for:
  - Age
  - Log_Income
- Outliers removed for robust statistical testing

---

## 📊 Statistical Analysis & Hypothesis Testing

### ✅ Hypothesis 1: Age vs In‑Store Purchases

- **Test:** Independent T‑test
- **Finding:** Older individuals show higher in‑store purchase behavior

### ✅ Hypothesis 2: Age × Channel Preference

- **Test:** Two‑Way ANOVA
- **Factors:** Age_Category × Channel (Store vs Web)
- **Output:**
  - ANOVA table
  - Interaction effect line plot
  - F‑distribution visualization

### ✅ Hypothesis 3: Children vs Online Purchases

- **Test:** Independent T‑test
- **Finding:** Customers with children make more web purchases

### ✅ Hypothesis 4: Store vs Alternative Channels

- **Test:** Spearman Correlation
- **Finding:** Examines channel cannibalization risk

### ✅ Hypothesis 5: US vs Rest of World Purchases

- **Test:** Independent T‑test
- **Note:** Small sample warning handled and documented

---

## 📈 Visual Analytics (Plotly)

Generated interactive charts include:

- Histograms + Box plots (with and without outliers)
- Correlation heatmaps
- Two‑Way ANOVA interaction plots
- F‑distribution plots
- Product revenue bar charts
- Campaign acceptance by country
- Children vs total spend trends
- Complaint count by education
- Total Revenue by Country through Map Plot
- Campaign Acceptance Rate by Country through Map Plot
- Total Purchases by Country through Map Plot

All plots are embedded into the **HTML report**.

---

## 📝 Key Insights Summary

- Income required imputation based on demographic similarity
- Outlier removal materially improves correlation clarity
- Older customers show stronger in‑store preference
- Families with children demonstrate higher web activity
- Product revenue concentration varies significantly
- Complaints show education-level patterns
- Certain tests limited by small sample sizes (documented)

---

## 📄 Report Output

The script generates an **interactive HTML report**: `reports/marketing_campaign_report.html`

- `reports/marketing_campaign_executive_dashboard.html`

Features:

- Collapsible data tables
- Statistical result cards
- Embedded Plotly charts
- Automatically opens in browser

---

## ▶️ How to Run

```bash
py -m Module-2.Projects.Capstone-1.marketing_campaign
```
