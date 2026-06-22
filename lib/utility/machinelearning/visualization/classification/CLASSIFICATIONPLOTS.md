# Classification Visualization Plots Documentation

## ✅ Overview

This module provides **classification-specific visualization utilities** for model evaluation results.

It is designed to work with:

- ✅ `Metrics` (evaluation layer)
- ✅ `MetricResolver` (metric selection + direction)
- ✅ `VisualizerEngine` (orchestration layer)

The module focuses on:

- ✅ Metric comparison
- ✅ Model ranking and highlighting
- ✅ Multi-metric visualization
- ✅ ROC curve visualization (binary + multiclass)

---

## ✅ Architecture

```
VisualizerEngine
      ↓
ClassificationPlots
      ↓
Plotly Visualizations
```

---

## ✅ Dependencies

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lib.utility.machinelearning.evaluation.MetricResolver import MetricResolver
from lib.utility.machinelearning.shared.DataCleaner import DataCleaner
```

---

## ✅ Internal Helpers

### `_filter()`

Filters dataframe by model, mode, and type.

### `_get_metrics()`

Selects default metrics using `MetricResolver` when not provided.

---

## ✅ Plot Functions

---

### ✅ 1. Scatter Plot

```python
plot_scatter(df, x_metric=None, y_metric=None, size=None, color="model")
```

- Automatically selects metrics if not provided
- Useful for performance relationship analysis

---

### ✅ 2. Bar Plot (Metric Comparison)

```python
plot_bar(df, metric=None, group_by="model")
```

- Uses `MetricResolver.get_best_metric()` if metric not provided
- Shows performance comparison across models/experiments

---

### ✅ 3. Best Model Highlight

```python
plot_best_model(df, metric=None)
```

- Uses metric direction (maximize/minimize)
- Highlights best model using annotation

---

### ✅ 4. Multi-Metric Comparison

```python
plot_multi_metrics(df, metrics=None)
```

Default metrics:

- accuracy
- f1_weighted
- f1_macro
- roc_auc

---

### ✅ 5. ROC Curve Comparison

```python
plot_roc_all_models(artifacts)
```

Supports:

- ✅ Binary classification
- ✅ Multiclass (macro averaging)

Includes:

- Best model highlighting
- Random baseline

---

## ✅ Auto Entry (Used by VisualizerEngine)

```python
plot_all(results, artifacts=None)
```

Returns:

```python
{
    "scatter": fig,
    "bar": fig,
    "multi_metric": fig,
    "best_model": fig,
    "roc_curve": fig
}
```

---

## ✅ Design Highlights

- ✅ Stateless functional design
- ✅ MetricResolver-driven
- ✅ Compatible with artifacts layer
- ✅ Supports binary, multiclass, multilabel
- ✅ Plug-and-play with VisualizerEngine

---

## ✅ Best Practices

- Always pass:
  - `results` → scalar metrics
  - `artifacts` → ROC/PR/confusion matrix
- Avoid mixing raw data with processed metrics
- Use experiment-level grouping instead of model-only grouping

---

## ✅ Summary

This module provides a **modular, extensible, and production-ready visualization layer** for classification tasks.

It ensures:

- ✅ Consistent evaluation visuals
- ✅ Metric-aware plotting
- ✅ Seamless integration with pipeline
