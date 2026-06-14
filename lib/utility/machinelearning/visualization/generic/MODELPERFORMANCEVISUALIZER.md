# ModelPerformanceVisualizer - Comprehensive Documentation

## 📌 Overview

The `ModelPerformanceVisualizer` is a **production-grade, experiment-aware visualization engine** designed for modern ML systems.

It supports:

- ✅ Regression metrics (R2, MSE, RMSE)
- ✅ Classification metrics (accuracy, F1, etc.)
- ✅ Hyperparameter tuning outputs
- ✅ Experiment-level tracking
- ✅ AutoML-ready visualization

---

# 🚀 Evolution of the System

### ❌ Before

- Model-level plotting
- Static metrics
- No tuning support

### ✅ Now

- Experiment-level visualization
- Dynamic metric detection
- Hyperparameter-aware plots
- Optimization tracking

---

# 🧩 Data Schema

Expected DataFrame structure:

| Column        | Description                  |
| ------------- | ---------------------------- |
| model         | Model name                   |
| experiment    | Unique experiment identifier |
| mode          | train-test / k-fold / tuning |
| type          | baseline / tuned             |
| R2, MSE, etc. | metrics                      |
| score         | tuning metric                |
| param\_\*     | hyperparameters              |

---

# 🧠 Core Components

### ✅ DataCleaner

- Handles NaN values
- Ensures numeric consistency
- Prevents visualization crashes

### ✅ MetricResolver

- Auto-detects suitable metrics
- Supports regression + classification

---

# 📊 Core Visualization Features

## ✅ Scatter Plot

- Auto metric selection
- Experiment-level grouping
- Handles missing values

## ✅ Bar Plot

- Model and experiment comparison

## ✅ Best Model Highlight

- Highlights top experiment

## ✅ Auto Plot

- Smart selection of metrics and plot type

---

# 🔬 Hyperparameter Visualization Integration

The visualizer works seamlessly with:

### ✅ HyperparameterPlots

- 2D plots (single parameter)
- 3D surfaces (multi-parameter)

### ✅ OptimizationPlots

- Iteration animation
- Optimization tracking

### ✅ ComparisonPlots

- Model vs experiment comparison

---

# ⚙️ Intelligent Behavior

## ✅ Dynamic Metric Detection

Priority:

1. score (tuning)
2. R2
3. other metrics

---

## ✅ Adaptive Plot Selection

| Scenario  | Plot Type          |
| --------- | ------------------ |
| 1 param   | 2D                 |
| 2+ params | 3D                 |
| baseline  | scatter/bar        |
| tuning    | optimization plots |

---

# 📦 Example Usage

```python
viz = ModelPerformanceVisualizer()
viz.auto_plot(results_df)
```

---

# 🔗 Integration Flow

```
LinearModelUtility
        ↓
HyperparameterTuner
        ↓
DataCleaner
        ↓
Visualizer
```

---

# ✅ Best Practices

- Always include `experiment` column
- Flatten hyperparameters
- Use `score` for tuning
- Avoid nested dicts

---

# 🚀 Advanced Capabilities

- ✅ AutoML-ready visualization
- ✅ Experiment comparison dashboard ready
- ✅ Multi-model + multi-config tracking

---

# 🔮 Future Enhancements

- Interactive dashboards (Streamlit)
- AutoML pipelines
- Experiment tracking UI
- SHAP integration

---

# 🏁 Conclusion

This system represents a **modern ML visualization framework** supporting:

✔ Experiment-level analysis
✔ Hyperparameter optimization
✔ AutoML workflows

---
