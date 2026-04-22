# DataFrame Utilities

This folder contains **reusable Pandas DataFrame utilities** designed to support
data loading, inspection, transformation, and presentation across the project.

The utilities in this module act as a **clean abstraction layer on top of Pandas**,
helping keep analysis scripts and report generation code:
- readable
- DRY
- consistent
- production‑oriented

---

## 📁 Folder Structure
```
lib/utility/dataframe/
│
├── data_loader.py       # Dataset loading & optimization helpers
├── df_helper.py         # DataFrame inspection & transformation helpers
├── init.py
```
---

## 🎯 Design Philosophy

These utilities are designed to:

- Encapsulate **common Pandas workflows**
- Separate **data handling logic** from analysis logic
- Provide **human‑readable summaries** for HTML reports
- Support **large datasets and teaching‑oriented output**
- Reduce boilerplate in scripts and notebooks

They are intentionally **stateless and functional**, making them easy to reuse.

---

## 📦 data_loader.py

### Purpose

`DataLoader` provides a **standardized and optimized way to load datasets**
into Pandas DataFrames.

It abstracts away repetitive logic such as:
- reading CSV files
- dropping unnamed columns
- memory optimization
- dataset quality reporting

---

### Key Responsibilities

- Read tabular datasets (CSV)
- Clean unwanted columns (e.g. `"Unnamed: 0"`)
- Optimize DataFrame memory usage
- Return structured metadata about the load operation

---

### Typical Usage

```python
from lib.utility.dataframe.data_loader import DataLoader as dl

df, report = dl.read_dataset(
    "housing_data.csv",
    optimize=True,
    handle_unnamed="drop",
    return_report=True
)
```
## 📦 df_helper.py

DataFrameHelper contains high‑level helper functions for inspecting,
summarizing, and transforming Pandas DataFrames in a report‑friendly way.
It focuses on:
- readability
- reproducibility
- structured output


### Key Capabilities

##### 1️⃣ DataFrame Structure Inspection
- get_dataframe_info_str()..
- Returns a string‑based version of df.info().

```Python
df_info_str = dfh.get_dataframe_info_str(df)
```
✅ Useful for:

HTML reports
logging
teaching outputs


#### 2️⃣ Row‑Wise Rendering Helpers
- dataframe_rows_as_pre(...)
- Converts DataFrame rows into formatted friendly strings.

```Python
df_str = dfh.dataframe_rows_as_pre(    df,    method="iterrows",    include_index=True,    index_label="Employee ID")
```
✅ Ideal for:

large datasets
row‑level inspection
readable report sections


#### 3️⃣ Column Manipulation Utilities
- insert_column_after(...)
- Adds a new column at a specific position, not just at the end.
```Python
df = dfh.insert_column_after(    df,    after_col="YearBuilt",    new_col="HouseAge",    values=age_series,    inplace=True)
```

✅ Solves a common Pandas pain point
✅ Preserves logical column ordering

## Why These Helpers Matter

- Pandas is powerful but verbose
- Reports need structured, readable output
- Teaching/learning requires clarity over cleverness

These helpers bridge the gap between raw Pandas and explainable analytics.

## 🧠 How These Modules Are Used Together
A typical workflow looks like:

```
DataLoader
   ↓
Optimized DataFrame
   ↓
DataFrameHelper
   ↓
Readable summaries & transformations
   ↓
HTML Reports / Analysis Scripts
```
They are frequently used alongside:

- HtmlBuilder
- PlotRenderer
- FormulaRegistry


## ✅ Benefits of This Utility Layer


| Problem                             | Solution                          |
|------------------------------------|-----------------------------------|
| Repeated Pandas boilerplate         | Central helper utilities          |
| Poor report readability             | String‑based structured output    |
| Uncontrolled column order           | Explicit insertion helpers        |
| Large dataset inspection issues     | Formatted + collapsible views     |
| Mixed logic & presentation          | Clean separation of concerns      |
