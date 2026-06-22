# RegressionPlots Documentation

## ✅ Overview

This module provides visualization utilities specific to **regression model evaluation**.

It focuses on comparing regression performance metrics such as:

- ✅ R² (Coefficient of Determination)
- ✅ RMSE (Root Mean Squared Error)

The module is designed to:

- ✅ Support experiment-level visualization
- ✅ Provide fallback to model-level grouping
- ✅ Work seamlessly with `VisualizerEngine`
- ✅ Handle missing or incomplete data safely

---

## ✅ Architecture

```
VisualizerEngine
      ↓
RegressionPlots
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

Determines grouping column:

- ✅ Uses `experiment` if available (preferred)
- ✅ Falls back to `model`

---

## ✅ Main Function

### ✅ `plot_all(results)`

```python
def plot_all(results):
```

### 🔹 Behavior

- Converts results into DataFrame
- Validates data availability
- Dynamically resolves grouping column
- Safely handles missing metrics

---

### ✅ Output Structure

```python
{
    "r2": fig,
    "rmse": fig
}
```

---

## ✅ Plot Details

---

### 📊 R² Plot

- Shows how well the model fits the data
- Higher is better (closer to 1)

```python
px.bar(
    df,
    x=group_col,
    y="R2",
    color=group_col
)
```

---

### 📊 RMSE Plot

- Measures prediction error
- Lower is better

```python
px.bar(
    df,
    x=group_col,
    y="RMSE",
    color=group_col
)
```

---

## ✅ Error Handling

| Scenario     | Behavior                   |
| ------------ | -------------------------- |
| No data      | Returns empty plots        |
| Missing R2   | Shows "R2 not available"   |
| Missing RMSE | Shows "RMSE not available" |

---

## ✅ Design Highlights

- ✅ Experiment-first visualization
- ✅ Robust against missing metrics
- ✅ Clean and minimal design
- ✅ Compatible with pipeline output
- ✅ Easy integration into dashboard

---

## ✅ Example Usage

```python
figs = plot_all(results)

figs["r2"].show()
figs["rmse"].show()
```

---

## ✅ Best Practices

- Always use experiment-level data for richer insights
- Combine with ranking plots for better interpretation
- Use residual plots alongside RMSE for deeper analysis

---

## ✅ Summary

This module provides a **simple, reliable, and production-ready** visualization layer for regression models.

It ensures:

- ✅ Clear comparison of model performance
- ✅ Consistent visualization across experiments
- ✅ Seamless integration with the ML framework
