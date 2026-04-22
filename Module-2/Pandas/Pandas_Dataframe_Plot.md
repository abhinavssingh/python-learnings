# Pandas Data Visualization using Plotly – HTML Report

This script demonstrates **data visualization techniques using Pandas-compatible datasets and Plotly**, and generates a **comprehensive interactive HTML report** containing multiple chart types.

It serves as a **visual analytics showcase**, illustrating how raw data can be transformed into rich, interactive insights using modern Python visualization libraries.

---

## 📌 Purpose of This Script

The main objectives of this script are to:

- Demonstrate **multiple chart types** using Plotly Express and Plotly Graph Objects
- Work with **real-world sample datasets** (Gapminder, Iris, Tips, Wind)
- Show how Pandas-style data can be visualized interactively
- Combine multiple plots into a **single, structured HTML report**
- Provide a reusable visualization reporting workflow

---

## 📊 Visualizations Included

This script generates a wide variety of charts, grouped by use case:

---

### 1. Line Charts
- **Life Expectancy in Canada over Time**
- **Life Expectancy Trends in Oceania by Country**

Used to visualize:
- Time series trends
- Country- and continent-level comparisons

---

### 2. Bar Charts
- Population of Canada by year
- Population of Canada colored by life expectancy
- Horizontal population comparison

Demonstrates:
- Standard and horizontal bars
- Color encoding for additional metrics
- Hover-based data enrichment

---

### 3. Area Chart
- **Population Growth by Continent Over Time**

Illustrates:
- Stacked area visualization
- Growth comparison across continents

---

### 4. Pie Charts
- European population distribution (2007)
- Tip distribution by day

Shows:
- Proportion-based analysis
- Aggregated category comparison

---

### 5. Scatter Plots
- Iris dataset: sepal length vs sepal width
- Color- and size-encoded scatter plots

Demonstrates:
- Multi-dimensional data visualization
- Clustering and differentiation by category

---

### 6. Box, Histogram & Violin Plots
- Box plot of total bill by day and smoker status
- Histogram of total bill by gender
- Violin plot for tip distribution

Highlights:
- Distribution analysis
- Outliers and data spread
- Category-based comparison

---

### 7. Treemap (Hierarchical Heatmap)
- Global population and life expectancy (2007)

Demonstrates:
- Hierarchical data representation
- Size + color encoding
- Weighted midpoint color scaling

---

### 8. Polar Charts
- Wind frequency by direction and strength
- Advanced categorical polar plots using subplots

Illustrates:
- Radar-style visualization
- Angular and radial category plotting
- Complex subplot layouts

---

## 🧱 Technical Components Used

| Component | Purpose |
|---------|--------|
| `plotly.express` | High-level interactive chart creation |
| `plotly.graph_objects` | Advanced, customizable plots |
| `make_subplots` | Multi-chart polar layouts |
| `HtmlBuilder` | Builds structured HTML pages |
| `PlotRenderer` | Embeds Plotly charts into HTML cards |
| `ReportUtils` | Saves and auto-opens the HTML report |

---

## 🖥 HTML Report Output

All charts are assembled into a **responsive, card-based HTML report**.

### Output File
**Generated File:**
 `reports/Pandas Data Visulaization through Various chart.html`
 ---
 ## 🛠 Utilities Used

| Utility | Purpose |
|------|--------|
| `HtmlBuilder` | Structured HTML output |
| `PlotRenderer` | Structured Plot output  |
| `ReportUtils` | Save & open HTML reports |

These utilities ensure **clear separation** between:
- Analysis
- Presentation

---