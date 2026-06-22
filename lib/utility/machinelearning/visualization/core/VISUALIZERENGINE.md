
# VisualizerEngine Documentation

## ✅ Overview

The `VisualizerEngine` is the **central orchestration layer** of the visualization module.

It is responsible for:
- ✅ Routing visualization requests by task (classification, regression, unsupervised)
- ✅ Integrating with `MetricResolver` for metric selection
- ✅ Delegating rendering to generic and task-specific plot modules
- ✅ Producing a unified dashboard output structure

---

## ✅ Architecture

```
Results + Artifacts
        ↓
VisualizerEngine
        ↓
├── Generic Plots (comparison, ranking, distribution)
├── Task-Specific Plots
│     ├── ClassificationPlots
│     ├── RegressionPlots
│     └── ClusteringPlots
```

---

## ✅ Dependencies

```python
import pandas as pd

from lib.utility.machinelearning.evaluation.MetricResolver import MetricResolver

from ..classification.ClassificationPlots import plot_all as classification_plots
from ..generic.ComparisonPlots import (
    plot_all_model_comparison,
    plot_best_model_highlight,
    plot_model_ranking,
)
from ..generic.DistributionPlots import plot_metric_distribution
from ..regression.RegressionPlots import plot_all as regression_plots
from ..unsupervised.ClusteringPlots import plot_metrics as clustering_plots
```

---

## ✅ Initialization

```python
engine = VisualizerEngine(results, artifacts)
```

### Inputs

| Parameter | Description |
|----------|------------|
| results | List of model results (metrics) |
| artifacts | Additional artifacts (ROC, PR, clusters) |

### Internal State

- `self.results` → raw results
- `self.df` → converted DataFrame (for plotting)
- `self.artifacts` → auxiliary data (ROC, clusters, etc.)
- `self.task` → inferred task type

---

## ✅ Generic Visualizations

### 1. Model Comparison

```python
engine.plot_comparison()
```

- Uses best metric via `MetricResolver`
- Delegates to `plot_all_model_comparison`

---

### 2. Model Ranking

```python
engine.plot_ranking()
```

- Produces rank-based comparison
- Metric direction aware (↑ / ↓)

---

### 3. Best Model Highlight

```python
engine.plot_best_model()
```

- Highlights best model based on metric direction

---

### 4. Metric Distribution

```python
engine.plot_distribution()
```

- Shows distribution of selected metric across experiments

---

## ✅ Task-Specific Visualizations

### Classification

```python
classification_plots(self.df, self.artifacts)
```

Includes:
- ROC curves
- Multi-metric comparison
- Scatter plots

---

### Regression

```python
regression_plots(self.df)
```

Includes:
- R2 comparison
- RMSE comparison

---

### Unsupervised

```python
clustering_plots(self.df)
```

Includes:
- Clustering metrics visualization

---

## ✅ Unified Dashboard Output

```python
engine.render_all()
```

### Output Format

```python
{
    "comparison": fig,
    "ranking": fig,
    "best_model": fig,
    "distribution": fig,
    "task_specific": {
        "plot1": fig,
        "plot2": fig
    }
}
```

---

## ✅ Design Highlights

- ✅ Centralized orchestration layer
- ✅ Task-aware visualization routing
- ✅ MetricResolver integration for consistency
- ✅ Clean separation of concerns
- ✅ Plug-and-play architecture
- ✅ Compatible with all ML paradigms

---

## ✅ Best Practices

- Always pass:
  - `results` → structured model results
  - `artifacts` → additional evaluation data

- Ensure `task` field exists in results

- Use experiment-based grouping instead of model-only grouping

---

## ✅ Summary

`VisualizerEngine` provides a **unified, extensible, and production-grade visualization layer**.

It bridges:
- evaluation (metrics + ranking)
- visualization (plot rendering)
- reporting (dashboard integration)

making it the **core visualization backbone of the ML framework**.

