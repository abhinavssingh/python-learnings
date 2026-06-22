
# DistributionPlots Documentation

## ✅ Overview

This module provides **distribution-focused visualizations** for machine learning workflows.

It includes utilities for:

- ✅ Metric distribution (model/experiment level)
- ✅ Residual analysis (regression)
- ✅ Class distribution (classification)
- ✅ Cluster distribution (unsupervised)

These plots are critical for:

- Debugging model performance
- Understanding data imbalance
- Evaluating clustering outputs
- Supporting explainability and reporting

---

## ✅ Architecture

```
VisualizerEngine
      ↓
DistributionPlots
      ↓
Plotly Visualizations
```

---

## ✅ Dependencies

```python
import pandas as pd
import plotly.express as px
```

---

## ✅ Internal Helper

### `_resolve_group_col(df)`

Determines grouping column for visualization:

- ✅ Uses `experiment` (preferred)
- ✅ Falls back to `model`

---

## ✅ Plot Functions

---

### ✅ 1. Metric Distribution

```python
plot_metric_distribution(results, metric)
```

- Converts results to DataFrame
- Groups by experiment/model
- Displays histogram of selected metric

---

### ✅ 2. Residual Distribution (Regression)

```python
plot_residual_distribution(y_true, y_pred)
```

- Computes residuals: `y_true - y_pred`
- Useful for:
  - Checking bias
  - Detecting outliers

---

### ✅ 3. Class Distribution (Classification)

```python
plot_class_distribution(y)
```

- Displays frequency of target classes
- Useful for imbalance analysis

---

### ✅ 4. Cluster Distribution (Unsupervised)

```python
plot_cluster_distribution(labels)
```

- Shows distribution of assigned cluster labels
- Helps detect:
  - dominant clusters
  - noise clusters (e.g., DBSCAN)

---

## ✅ Design Highlights

- ✅ Lightweight and focused
- ✅ Supports all ML paradigms
- ✅ Experiment-aware visualization
- ✅ Stateless functional design
- ✅ Works seamlessly with Plotly

---

## ✅ Best Practices

- Use experiment grouping over model-only grouping
- Combine with comparison plots for deeper insights
- Use residual plots to validate regression assumptions
- Always validate metric availability before plotting

---

## ✅ Example Usage

```python
fig = plot_metric_distribution(results, metric="f1_weighted")
fig.show()
```

---

## ✅ Summary

The `DistributionPlots` module provides essential tools for **understanding the spread and behavior of metrics and outputs**.

It enhances:

- ✅ interpretability
- ✅ debugging
- ✅ reporting quality

and acts as a key complement to comparison and ranking visualizations.

