# ML Linear Regression Pipeline Report Script

This document explains the end-to-end machine learning pipeline used for building, evaluating, tuning, and visualizing linear regression models.

---

## Overview

This script:

- Loads and preprocesses a marketing dataset
- Applies imputation and outlier handling
- Trains multiple linear models
- Runs experiments (K-Fold, custom configs)
- Performs hyperparameter tuning (Grid & Random Search)
- Generates rankings and comparisons
- Builds an interactive HTML report

---

## Key Components

### Imports

```python
import numpy as np
import pandas as pd
```

Custom utilities used:

- HtmlBuilder → HTML layout builder
- PlotRenderer → Plotly rendering helper
- DataLoader → dataset loading
- DataFrameHelper → dataframe utilities
- CustomImputer → missing value handling
- OutlierHandler → outlier treatment
- LinearModelUtility → ML pipeline core
- ModelPerformanceVisualizer → visualization layer
- ReportUtils → saving reports

---

## Data Preparation

Steps performed:

1. Load dataset (`marketing_data.csv`)
2. Clean `Income` column (remove `$` and commas)
3. Convert `Dt_Customer` to datetime
4. Create `TotalSpend` feature (sum of all `Mnt` columns)
5. Insert new column using helper utility

---

## Preprocessing

- Missing values handled using `CustomImputer`
- Outliers handled using `OutlierHandler (IQR method)`

---

## Model Training

### Default Training

```python
ml_results = lm.run_all_models()
```

### K-Fold Validation

```python
ml_kfold_results = lm.run_experiment(model_name="LinearRegression", k_fold=5)
```

### Custom Experiments

```python
configs = [...]
ml_selected_results = lm.run_experiments(configs)
```

---

## Hyperparameter Tuning

### Grid Search (Ridge)

```python
param_grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
ridge_grid_result = lm.grid_search_cv(...)
```

### Tuned Models

- Ridge → Grid Search
- ElasticNet → Random Search

---

## Evaluation

Generated artifacts:

- Model ranking
- Best model
- Comparison summary
- Combined results dataframe
- Improvement vs baseline

---

## Visualization

Using `ModelPerformanceVisualizer`:

- Train vs K-Fold comparison
- Best model annotation
- Optimization animation
- 3D hyperparameter surface
- GridSearch animation

---

## Report Generation

HTML report built using:

```python
html_doc = builder.build_page(...)
```

Saved using:

```python
ru.save_html_report(...)
```

---

## Output

- Interactive HTML report
- Model insights and comparisons
- Tuning behavior visualization
