
# ClassificationPlots – Detailed Documentation

## Overview

The `ClassificationPlots` class provides visualization utilities for classification model evaluation. It is built on top of Plotly and supports binary, multiclass, and multilabel outputs.

---

## Key Features

- Metric comparison across models
- ROC curve visualization (multi-model support)
- Scatter and bar plots
- Multi-metric visualization
- Auto metric selection
- Supports multilabel aggregation

---

## Architecture

```
Results DataFrame → DataCleaner → MetricResolver → Plotly → Visualization
```

---

## Filtering

```python
_filter(df, model=None, mode=None, type_=None)
```

Filters dataset based on:
- model
- mode (train/test)
- type (baseline/tuned)

---

## Scatter Plot

```python
plot_scatter(df, x_metric=None, y_metric=None)
```

### Features
- Auto metric selection
- Optional size scaling
- Color grouping by model

---

## Bar Plot

```python
plot_bar(df, metric="accuracy")
```

### Features
- Compare models on selected metric
- Grouping by model

---

## Best Model Visualization

```python
plot_best_model(df, metric="f1")
```

### Features
- Highlights best-performing model
- Annotates best score

---

## Multi-Metric Plot

```python
plot_multi_metrics(df, metrics=[...])
```

### Features
- Compare multiple metrics
- Grouped bar visualization

---

## ROC Curve (Advanced)

```python
plot_roc_all_models(results)
```

### Binary
- Plots ROC curve
- Highlights optimal threshold

### Multiclass
- One-vs-Rest curves
- Aggregated visualization

### Multilabel
- Macro-averaged ROC
- Handles per-label curves

### Additional Features
- Best model highlighting
- Random baseline line

---

## Auto Metric Selection

```python
_default_metrics(df)
```

Priority:
1. roc_auc
2. f1
3. accuracy

---

## Auto Plot

```python
auto_plot(df)
```

Generates scatter plot with best metrics automatically.

---

## Dependencies

- Plotly (px, go)
- DataCleaner
- MetricResolver

---

## Design Principles

- Clean visualization separation
- Data validation before plotting
- Flexible across problem types
- Scalable for dashboards

---

## Best Practices

- Ensure clean results DataFrame
- Avoid plotting raw artifacts
- Use formatter for artifact rendering

---

## Extensibility

- Precision-Recall curves
- Confusion matrix heatmaps
- Per-label visualization
- Interactive dashboards

---

## Summary

`ClassificationPlots` provides a powerful and extensible visualization layer for ML evaluation pipelines, supporting both simple and advanced model comparisons.

