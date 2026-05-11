# Sales Analytics Callout & Visualization Justification

## 1. Explicit Peak and Off‑Peak Sales Callout

### Time‑of‑Day Sales Performance Analysis

This analysis identifies peak and off‑peak sales periods using time‑of‑day histograms and heatmaps generated from cleaned sales data.

#### Peak Sales Periods

- **Morning hours (approximately 9 AM – 12 PM)** consistently show higher sales volumes across most states.
- Sales during this period are particularly strong for **Men and Seniors** customer groups.
- Indicates higher customer purchase intent and footfall earlier in the day.

#### Off‑Peak Sales Periods

- **Late afternoon (3 PM – 5 PM)** shows lower sales in the month of **October**.
- **Evening hours (after 6 PM)** show reduced sales during **November and December**.
- Suggests customer fatigue, reduced footfall, or seasonal behavioral changes.

#### Business Implications

Sales and Marketing (S&M) teams can:

- Schedule promotions, hyper‑personalized offers, and sales staffing during **peak morning hours**.
- Boost off‑peak conversion through targeted incentives such as flash discounts, personalized notifications, or bundled offers.
- Use time‑based insights to optimize workforce allocation and campaign timing.

---

## 2. Justification of Visualization Choices (Why These Charts?)

### KPI Summary Table (Plotly Table)

- Summarizes key metrics such as Total Sales, Average Sales, Median Sales, and Data Cleaning Impact.
- Chosen instead of indicator tiles to avoid text overlap and improve readability.
- Provides numerical values alongside business explanations, enabling quicker executive interpretation.
- In card view not displaying full details. Click on View Details to get full details.

**Best suited for:** Executive dashboards and summary-level reporting.

---

### Box Plots (With and Without Outliers)

- Used to visualize data distribution and identify extreme values.
- Effectively demonstrates the impact of outliers on Sales and Units.
- Supports the decision to apply log transformation and IQR-based outlier removal.

**Best suited for:** Distribution comparison and data quality validation.

---

### Pie Charts (State-wise Sales Contribution)

- Displays the proportional contribution of each state to overall sales.
- Helps identify high-performing and underperforming regions.
- Useful for geographical sales strategy and resource allocation.

**Best suited for:** Percentage-based contribution analysis.

---

### Sunburst Charts (Hierarchical Sales Distribution)

- Represents multi-level sales distribution across:
  **State → Month → Week → Day**
- Enables interactive drilling into specific time periods or regions.
- Provides a clear storytelling structure for management review.

**Best suited for:** Hierarchical and multi-dimensional analysis.

---

### Histograms (Daily, Monthly, Quarterly)

- Show frequency distributions and seasonal patterns.
- Useful for understanding sales consistency and variability over time.
- Highlight peak and slow sales periods across different calendar intervals.

**Best suited for:** Temporal trend and seasonality analysis.

---

### Heatmaps (State × Group, Time × Group, Month × Time)

- Enable comparison of sales intensity across two categorical dimensions.
- Color gradients quickly highlight peak and off‑peak zones.
- Highly effective for identifying customer behavior patterns.

**Best suited for:** Comparative analysis across multiple dimensions.

---

## 3. Why Plotly Was Used Instead of Seaborn

Although Seaborn is effective for static statistical plots, Plotly was chosen for this analysis due to the following advantages:

- Interactive features such as hover tooltips, zooming, and drill‑downs.
- Seamless integration with HTML dashboards.
- Better suited for executive and stakeholder presentations.
- Scales well with larger datasets.
- Enhances data storytelling for business decision‑making.

**Justification Statement:**

> _Although Seaborn is well-suited for statistical exploration, Plotly was selected to build the dashboard due to its interactivity, scalability, and executive-level readability._

---

## 4. Summary

- The selected charts translate statistical results into actionable business insights.
- Time-of-day analysis identifies clear peak and off-peak sales periods.
- Visualizations were chosen based on clarity, interpretability, and relevance to Sales & Marketing leadership.
- The dashboard design supports data-driven strategic decision-making.

** Answer:**

> _“The visualizations were selected to convert statistical analysis into executive-level insights, enabling clear identification of trends, peak performance periods, and strategic opportunities.”_
