# ModelPerformanceVisualizer Documentation

This markdown file documents the `ModelPerformanceVisualizer` class used for visualizing machine learning experiment results using Plotly.

---

## Overview

The `ModelPerformanceVisualizer` provides multiple plotting utilities for:

- Model comparison
- Hyperparameter tuning visualization
- Best model selection
- Optimization tracking
- Preprocessing impact

---

## Imports

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
```

---

## Class: ModelPerformanceVisualizer

### 1. plot_all_model_comparison

Compare all models using R2 and MSE.

### 2. plot_model_comparison

Compare selected models.

### 3. plot_best_models

Shows best configuration per model.

### 4. plot_mode_comparison

Compare modes like train-test vs cross-validation.

### 5. plot_preprocessing_impact

Shows impact of imputation and preprocessing choices.

### 6. plot_model_ranking

Displays ranking of models based on R2.

### 7. plot_optimization_animation

Animated visualization of optimization progress.

### 8. plot_hyperparameter_surface_3d

3D scatter plot for hyperparameter analysis.

### 9. plot_hyperparameter_3d_scatter

Alternative 3D visualization.

### 10. plot_best_model_highlight

Highlights best model globally.

### 11. plot_best_per_model_highlight

Highlights best configuration per model.

### 12. plot_best_with_annotation

Adds annotation for best model.

### 13. plot_gridsearch_animation

Animated visualization of GridSearch tuning.

### 14. get_flat_result

Utility to flatten nested result dictionaries.

---

## Notes

- Handles NaN values gracefully
- Supports filtering by model and mode
- Compatible with GridSearchCV outputs

---

## Usage Example

```python
viz = ModelPerformanceVisualizer()
fig = viz.plot_all_model_comparison(results_df)
fig.show()
```
