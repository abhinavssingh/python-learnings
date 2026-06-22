# 📊 Visualization Module README

## ✅ Overview

The **Visualization module** is the final layer in the ML framework responsible for transforming evaluation results into **interactive, interpretable, and dashboard-ready visual insights**.

It sits on top of:

```
Pipeline → Evaluation (Metrics + Comparator) → Visualization → Reporting
```

---

## ✅ Architecture

```
Results + Artifacts
        ↓
VisualizerEngine  (Core Orchestrator)
        ↓
├── Generic Plots
│     ├── ComparisonPlots
│     └── DistributionPlots
│
├── Task-Specific Plots
│     ├── ClassificationPlots
│     ├── RegressionPlots
│     └── ClusteringPlots
│
└── Advanced Exploration
      └── DimensionalityPlots (PCA, t-SNE)
```

---

## ✅ Core Components

### 🔹 1. VisualizerEngine (Core Layer)

Responsible for:

- Routing visualization based on task
- Selecting metrics using `MetricResolver`
- Combining generic + task-specific plots
- Producing unified dashboard output

```python
viz = VisualizerEngine(results, artifacts)
dashboard = viz.render_all()
```

---

### 🔹 2. Generic Visualization Layer

#### ✅ ComparisonPlots

- Model comparison (multi-metric, dual-axis)
- Ranking
- Best model highlight
- Preprocessing impact

#### ✅ DistributionPlots

- Metric distribution
- Residual distribution
- Class distribution
- Cluster distribution

---

### 🔹 3. Task-Specific Visualization

| Module              | Responsibility             |
| ------------------- | -------------------------- |
| ClassificationPlots | ROC, multi-metric, scatter |
| RegressionPlots     | R², RMSE                   |
| ClusteringPlots     | Cluster metrics            |

---

### 🔹 4. Dimensionality Visualization

#### ✅ DimensionalityPlots

- PCA (2D / 3D)
- t-SNE

Used for:

- cluster visualization
- high-dimensional data exploration

---

## ✅ Input Contract

All visualization components rely on two inputs:

### ✅ Results

Structured list of model outputs:

```python
{
    "model": "RandomForest",
    "experiment": "RF | kfold=5",
    "task": "classification",
    "accuracy": 0.92,
    "f1_weighted": 0.91
}
```

---

### ✅ Artifacts

Additional model outputs:

- ROC curves
- confusion matrix
- clustering labels
- embeddings

---

## ✅ Output Contract

`VisualizerEngine.render_all()` returns:

```python
{
    "comparison": fig,
    "ranking": fig,
    "best_model": fig,
    "distribution": fig,
    "task_specific": {
        "plot_name": fig
    }
}
```

---

## ✅ Design Principles

- ✅ Separation of concerns
- ✅ Experiment-first visualization (not just model)
- ✅ Metric-driven plotting (via MetricResolver)
- ✅ Stateless functional design
- ✅ Plug-and-play architecture
- ✅ Scalable across ML paradigms

---

## ✅ Supported ML Types

| Type           | Supported |
| -------------- | --------- |
| Classification | ✅        |
| Regression     | ✅        |
| Multi-class    | ✅        |
| Multi-label    | ✅        |
| Unsupervised   | ✅        |

---

## ✅ Example Usage (Pipeline Integration)

```python
viz = VisualizerEngine(results, artifacts)

dashboard = viz.render_all()

content.append(
    plotRenderer.plot_to_card(dashboard["comparison"], "Model Comparison")
)
```

---

## ✅ Dashboard Integration

Works with:

- ✅ HTML Builder (Tailwind UI)
- ✅ PlotRenderer
- ✅ Export to HTML report

---

## ✅ Best Practices

- Always include `experiment` in results
- Always pass artifacts separately
- Avoid hardcoding metrics
- Use MetricResolver for consistency

---

## ✅ Future Enhancements

- ✅ Toggle views (model vs experiment)
- ✅ Interactive filtering
- ✅ Explainability plots
- ✅ Auto dashboard templates

---

## ✅ Summary

The Visualization module provides a **complete, scalable, and production-ready system** for rendering insights across all machine learning workflows.

It ensures:

- ✅ consistency
- ✅ extensibility
- ✅ interpretability
- ✅ seamless integration with reporting

---

## 🚀 Final Thought

This module acts as the **presentation layer of your ML framework**, turning raw metrics into actionable insights.
