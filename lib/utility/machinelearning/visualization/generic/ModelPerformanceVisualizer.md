# ModelPerformanceVisualizer Documentation

## Overview

The `ModelPerformanceVisualizer` is a **generic, model-agnostic visualization utility** designed to support:

- Regression models (R2, MSE, RMSE)
- Classification models (accuracy, F1, etc.)
- Future extensibility (clustering, AutoML)

It adapts dynamically using:

- `DataCleaner` → handles missing values safely
- `MetricResolver` → detects and selects metrics automatically

---

## Class: ModelPerformanceVisualizer

### Purpose

Provides reusable and robust plotting utilities for ML results stored in a DataFrame.

---

## Internal Methods

### `_filter(df, model=None, mode=None)`

Filters data based on model name or execution mode.

**Parameters:**

- `df`: DataFrame
- `model`: optional model name
- `mode`: optional mode ("train-test", "k-fold", etc.)

**Returns:** Filtered DataFrame

---

## Public Methods

### 1. `plot_scatter()`

Creates a dynamic scatter plot.

**Features:**

- Works for regression and classification
- Automatically detects metrics if not provided
- Handles missing values safely

**Parameters:**

- `df`: input DataFrame
- `x_metric`: metric for x-axis
- `y_metric`: metric for y-axis
- `size`: optional bubble size
- `color`: grouping variable (default: model)
- `mode`: filter mode

**Example:**

```python
viz.plot_scatter(df, "MSE", "R2")
```

---

### 2. `plot_bar()`

Creates a bar chart for comparing models.

**Parameters:**

- `metric`: column to plot
- `group_by`: default = "model"
- `mode`: optional filter

**Example:**

```python
viz.plot_bar(df, "R2")
```

---

### 3. `plot_best_model()`

Highlights the best model based on a metric.

**Features:**

- Automatically selects the best row
- Adds annotation to plot

**Example:**

```python
viz.plot_best_model(df, "R2")
```

---

### 4. `auto_plot()`

Automatically detects metrics and generates a scatter plot.

**Example:**

```python
viz.auto_plot(df)
```

---

## Design Principles

### ✅ 1. Model-Agnostic

Supports any ML model type.

### ✅ 2. Metric-Agnostic

Metrics are dynamically detected, not hardcoded.

### ✅ 3. Robust Data Handling

Uses `DataCleaner` to:

- Drop invalid rows
- Handle NaN safely

### ✅ 4. Extensible Architecture

Works seamlessly with:

- ComparisonPlots
- HyperparameterPlots
- OptimizationPlots

---

## Dependencies

- pandas
- plotly.express
- DataCleaner
- MetricResolver

---

## Example Usage

```python
from ModelPerformanceVisualizer import ModelPerformanceVisualizer

viz = ModelPerformanceVisualizer()

# basic scatter
viz.plot_scatter(results_df, "MSE", "R2")

# auto plot
viz.auto_plot(results_df)

# bar chart
viz.plot_bar(results_df, "R2")

# best model
viz.plot_best_model(results_df, "R2")
```

---

## Notes

- Ensure metrics exist in DataFrame
- Ensure DataCleaner supports numeric only operations
- Flatten hyperparameters for integration with advanced plots

---

## Future Enhancements

- Auto plot selection by problem type
- Interactive dashboard integration (Streamlit / Dash)
- Experiment tracking support

---
