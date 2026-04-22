# Capstone‑1 Data Visualization – Interactive Analytics Report

This capstone project focuses on **large‑scale data visualization and exploratory data analysis (EDA)** using **Pandas, NumPy, and Plotly**.  
It transforms a real‑world dataset into meaningful insights through **interactive charts** and presents the results in a **self‑contained HTML report**.

The project simulates a **BI / analytics dashboard workflow**, similar to tools like Power BI or Tableau, but implemented fully in Python.

---

## 📌 Objective of This Capstone

The main goals of this project are to:

- Perform analytics‑ready data transformations
- Engineer categorical features using binning
- Analyze relationships between demographics, income, health, and region
- Create **interactive visualizations** across many chart types
- Summarize insights for decision‑making
- Generate a **professional HTML analytics report**

---

## 📂 Dataset Used

**File**
`NSMES1988.csv`

The dataset is loaded using a reusable `DataLoader` utility with:

- Automatic removal of unnamed columns
- Memory optimization
- A structured load/optimization report

---

## 🔄 Data Preparation & Feature Engineering

### 1️⃣ Numeric Scaling & Type Safety

- `age` scaled and converted to `float32`
- `income` scaled to USD and converted to `float32`

Why:
- `float16` is good for storage and ML tensors
- `float32` is safer for analytics, indexing, and binning

---

### 2️⃣ Categorical Binning

#### Age Categories
| Category | Range (years) |
|--------|---------------|
| Infant | 0–2 |
| Toddler | 2–4 |
| Kid | 4–13 |
| Teen | 13–20 |
| Adult | 20–60 |
| Senior | 60–75 |
| Super Senior | 75–100 |
| Ultra Senior | 100+ |

#### Income Categories
- **Low:** < 40K USD  
- **Medium:** < 80K USD  
- **High:** ≥ 80K USD  

Binning is implemented using `pd.cut()`.

---

### 3️⃣ Structured Column Insertion

New columns are inserted **after logical parent columns** using `DataFrameHelper`:

- `age category` after `age`
- `currency` after `income`
- `income category` after `currency`

This preserves column order and readability.

---

## 📊 Aggregations Performed

Grouped datasets are created for analysis, including:

- Age × Income
- Health × Age
- Age × Gender
- Health × Gender
- Income × Gender
- Region × Income
- Region × Health
- Multi‑factor category combinations

These aggregated views drive most of the visualizations.

---

## 📈 Visualizations Included

This capstone uses **Plotly** for fully interactive charts, including:

### 🔥 Heatmaps
- Age vs Income
- Region vs Income
- Health vs Age
- Health vs Gender
- Income vs Gender
- Correlation matrix (numeric features)

---

### 📉 Scatter & Scatter Matrix
- Visits vs Age (colored by region)
- Multi‑dimensional scatter matrices for visits, income, and age

---

### 📊 Bar Charts
- Standard bar charts
- Stacked bar charts
- Grouped bar charts

Used to compare:
- Age categories
- Income categories
- Regions
- Genders

---

### 📦 Histograms
- Region vs Insurance
- Region vs Medical Aid
- Region vs Employment
- Categorical count distributions
- Faceted histograms by gender

---

### 🧁 Pie Chart
- Regional distribution across categories

---

### 📦 Box & Violin Plots
- Distribution of visits and income across regions
- Comparison of spread and skewness

---

### 🌞 Sunburst & Treemap
- Hierarchical breakdowns by:
  - Region
  - Health
  - Gender
  - Insurance
  - Education

---

### 🔗 Parallel Categories
- Multi‑dimensional categorical flows across:
  - Region
  - Gender
  - Health

---

## 🧠 Key Insights Generated

Some of the insights derived from visual analysis include:

- Plotly enables PowerBI‑like interactive dashboards in Python
- Significant memory optimization (~13.44%)
- Data is dominated by female samples (~59.65%)
- Female health conditions are generally poorer than males
- Females earn less on average, possibly contributing to health outcomes
- Senior and super‑senior age groups dominate the dataset
- Midwest and “Other” regions require targeted health programs
- Recommendations made for:
  - Female health awareness
  - Insurance programs
  - Employment initiatives
  - Medical aid support

All insights are captured **inside the HTML report** for reference.

---

## 🖥 HTML Report Output

The script generates a **single comprehensive interactive report**.

**Output file**
`reports/capstone_1_Data_Visualization_report.html`

---

## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `DataLoader` | Dataset loading & optimization |
| `DataFrameHelper` | Column & DataFrame helpers |
| `HtmlBuilder` | Structured HTML layout |
| `PlotRenderer` | Embed Plotly charts |
| `ReportUtils` | Save & open reports |

---