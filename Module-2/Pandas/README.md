
# Module 2 – Pandas

This module contains Python scripts and examples created while learning **Pandas**, the most widely used Python library for data manipulation and analysis.

The focus of this module is to build a strong foundation in working with tabular data using `pandas`, covering everything from basic data structures to common data analysis operations.

---

## 📌 What You Will Learn

By going through the scripts in this folder, you will learn how to:

- Understand Pandas data structures (`Series` and `DataFrame`)
- Load and explore data from different sources
- Perform data selection, filtering, and indexing
- Handle missing data
- Manipulate and transform datasets
- Apply basic statistical and aggregation operations

---

## 📂 Folder Contents

This folder contains multiple Python scripts demonstrating Pandas concepts such as:

## Scripts

### `pandas_series_fundamentals.py`

The main goal of this script is to:

- Teach the basics of **Pandas Series**
- Demonstrate different ways to create a Series
- Showcase indexing, slicing, and filtering techniques
- Apply string operations on Series
- Generate an **HTML report** capturing all observations
- Saves: `reports/pandas_series_report.html`
---
### `pandas_series_maths.py`

The primary objectives of this script are to:

- Perform **element-wise mathematical operations** on Pandas Series
- Handle missing values (`NaN`) correctly
- Apply transformations and conditions on Series data
- Demonstrate conditional filtering techniques
- Generate a structured **HTML report** for visualization and learning
- Saves: `reports/pandas_series_maths_report.html`
---

### `pandas_series_exercise.py`
The primary objectives of this script are to:

- Practice real‑world operations on a Pandas `Series`
- Perform descriptive statistics on indexed data
- Demonstrate label‑based and pattern‑based selection
- Identify outliers using multiple statistical approaches
- Analyze day‑over‑day changes
- Generate a professional **HTML report** summarizing insights
- Saves: `reports/pandas_series_exercise_report.html`

### `pandas_dataframe_fundamentals.py`

The main goals of this script are to:

- Demonstrate creation of Pandas DataFrames from:
  - Dictionaries
  - Lists
  - NumPy arrays
  - Randomly generated data
  - Generate a **column-wise, card-based HTML report** summarizing DataFrame insights
  - Saves: `reports/pandas_dataframe_fundamentals_report.html`

---

### `pandas_dataframe_indexing_slicing.py`
The primary objective of this script is to:

- Explain **row and column selection** in Pandas DataFrames
- Compare `loc`, `iloc`, `at`, and `iat`
- Demonstrate **conditional filtering**
- Present results in a **structured HTML report**
- Saves: `reports/pandas_dataframe_indexing_slicing_report.html`

---

### `pandas_dataframe_maths.py`

The objective of this script is to:

- Perform **element‑wise mathematical operations** on DataFrames
- Apply **NumPy functions and lambda expressions** to DataFrames
- Demonstrate **DataFrame concatenation and merging**
- Explain **different join types** (inner, outer, left, right)
- Present all results in a **structured HTML report**
- Saves: `reports/pandas_dataframe_maths_report.html`

---

### `pandas_dataframe_dates.py`
The primary goals of this script are to:

- Generate a continuous date range using Pandas
- Enrich dates with **fiscal calendar fields (India-specific)**
- Perform calendar and fiscal aggregations
- Analyze weekends vs weekdays
- Optimize DataFrame memory usage
- Present all results in a **structured HTML report**
- Saves: `reports/pandas_date_range_dataframe_report.html`

### `pandas_dataframe_plot.py`
The main objectives of this script are to:

- Demonstrate **multiple chart types** using Plotly Express and Plotly Graph Objects
- Work with **real-world sample datasets** (Gapminder, Iris, Tips, Wind)
- Show how Pandas-style data can be visualized interactively
- Combine multiple plots into a **single, structured HTML report**
- Provide a reusable visualization reporting workflow
- Saves: `reports/Pandas Data Visulaization through Various chart.html`

### `pandas_dataframe_exercise.py`
The main objectives of this exercise are to:

- Load and optimize a real CSV dataset
- Reinforce **DataFrame axis fundamentals**
- Engineer new features using vectorized operations
- Perform descriptive statistics and correlation analysis
- Separate numeric and non‑numeric data intelligently
- Create an **interactive HTML report** for large datasets
- Saves: `reports/pandas_dataframe_exercise_report.html`


## Execution Modes

### 1. Direct Execution
`python pandas_series_fundamentals.py`

### 2. Module Execution
`python -m Module-2.Pandas.pandas_series_fundamentals`

### 3. Run Through Project Runner

```Python
Run all Pandas scripts:
python run.py --only "Module-2.Pandas.*"

or
Run a predefined list:
python run.py --config runlist.json
```

---

## 🛠 Prerequisites

Make sure you have the following installed:

- Python 3.x
- Pandas library

Install Pandas using:
```bash
pip install pandas
```