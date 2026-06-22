
# ComparisonPlots (Generic Visualization Layer)

## ✅ Overview

This module provides **generic, task-agnostic visualization functions** used for comparing, ranking, and analyzing machine learning models.

It is part of the core visualization layer and is designed to:

- ✅ Work across classification, regression, and unsupervised tasks
- ✅ Integrate with `MetricResolver` for metric selection and direction
- ✅ Support experiment-level visualization (preferred) with fallback to model-level
- ✅ Provide reusable and extensible plot utilities

---

## ✅ Architecture

```
VisualizerEngine
      ↓
ComparisonPlots (Generic Layer)
      ↓
Plotly Visualizations
```

---

## ✅ Dependencies

```python
import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.evaluation.MetricResolver import MetricResolver
from lib.utility.machinelearning.shared.DataCleaner import DataCleaner
```

---

## ✅ Internal Helpers

### `_resolve_group_col(df)`

Determines grouping column:

- ✅ Uses `experiment` (preferred)
- ✅ Falls back to `model`

---

### `_get_metric(df, metric, task=None)`

- Uses provided metric if valid
- Else resolves using `MetricResolver`
- Fallback: first numeric column

---

### `_get_direction(metric)`

Uses `MetricResolver` to determine optimization direction:

- `1` → maximize (e.g., accuracy, R2)
- `-1` → minimize (e.g., RMSE, DBI)

---

## ✅ Plot Functions

---

### ✅ 1. All Model Comparison

```python
plot_all_model_comparison(df, metric1=None, metric2=None, task=None)
```

- Dual-axis comparison
- Supports experiment-aware grouping

---

### ✅ 2. Selected Model Comparison

```python
plot_model_comparison(df, model_list, metric=None)
```

- Filters selected models
- Useful for focused analysis

---

### ✅ 3. Best Model Highlight

```python
plot_best_model_highlight(df, metric=None)
```

- Highlights best model using metric direction
- Automatically handles max/min metrics

---

### ✅ 4. Best Per Model

```python
plot_best_per_model(df, metric=None)
```

- Picks best configuration per model
- Uses direction-aware grouping

---

### ✅ 5. Mode Comparison

```python
plot_mode_comparison(df, metric=None)
```

- Compares modes like train/test vs k-fold

---

### ✅ 6. Preprocessing Impact

```python
plot_preprocessing_impact(df, x=None, y=None)
```

- Scatter plot analyzing preprocessing effects
- Supports imputer/outlier comparison

---

### ✅ 7. Model Ranking

```python
plot_model_ranking(df, metric=None)
```

- Sorts models based on metric
- Adds rank column (lower is better)

---

## ✅ Design Highlights

- ✅ Fully task-agnostic
- ✅ MetricResolver-driven (no hardcoding)
- ✅ Experiment-aware grouping
- ✅ Direction-aware logic (max/min)
- ✅ Stateless functional design

---

## ✅ Best Practices

- Use experiment column for richer insights
- Always pass cleaned results data
- Avoid hardcoding metric names
- Use `MetricResolver` consistently

---

## ✅ Output Example

```python
fig = plot_model_ranking(results_df, metric="f1_weighted")
fig.show()
```

---

## ✅ Summary

This module forms the **core comparison layer** of your visualization system.

It ensures:

- ✅ Consistent evaluation visuals
- ✅ Reusability across all ML tasks
- ✅ Clean integration with VisualizerEngine

