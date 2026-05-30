# LinearModelUtility Documentation

This document explains the `LinearModelUtility` class for building, evaluating, and tuning linear machine learning models using Scikit-learn.

---

## Overview

`LinearModelUtility` is a reusable, experiment-driven ML pipeline utility with support for:

- Data preprocessing
- Model training (train-test & k-fold)
- Hyperparameter tuning (Grid & Random search)
- Model comparison and ranking
- Baseline vs tuned performance tracking

---

## How It Works

The class automates the ML pipeline:

```
Raw Data
   ↓
Train-Test Split
   ↓
Preprocessing
   • Optional Imputer
   • Optional Outlier Handling
   • Scaling (StandardScaler)
   • Encoding (OneHotEncoder)
   ↓
Model Training
   ↓
Evaluation / Cross-Validation / Tuning
```

---

## Imports

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

---

## Initialization

```python
lm = LinearModelUtility(df, target_col="target", imputer=imputer, outlier_handler=outlier)
```

### Key Inputs

- `df`: Input DataFrame
- `target_col`: Target column name
- `imputer`: Optional missing value handler
- `outlier_handler`: Optional outlier handler

---

## Features

### 1. Data Preparation

- Splits data into train and test sets
- Detects numeric and categorical columns

```python
lm.prepare_data()
```

---

### 2. Preprocessing Pipeline

- Supports custom imputer & outlier handler
- Applies scaling and one-hot encoding

---

### 3. Model Training

#### Train-Test

```python
lm.run_experiment("Ridge")
```

#### K-Fold Validation

```python
lm.run_experiment("Ridge", k_fold=5)
```

---

### 4. Run Multiple Experiments

```python
configs = [
    {"model_name": "Ridge", "k_fold": 5},
    {"model_name": "Lasso"}
]

lm.run_experiments(configs)
```

---

### 5. Train All Models

```python
lm.run_all_models()
```

---

### 6. Grid Search

```python
param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
lm.grid_search_cv("Ridge", param_grid)
```

---

### 7. Hyperparameter Tuning

```python
lm.tune_model("ElasticNet", param_grid, search_type="random")
```

---

### 8. Ranking Models

```python
lm.rank_models("R2")
```

---

### 9. Best Model

```python
lm.get_best_model("R2")
```

---

### 10. Model Comparison

```python
lm.compare_models()
```

---

### 11. Combined Results DataFrame

```python
lm.get_combined_results_df()
```

Includes:

- rankings
- best flags
- tuning flags

---

### 12. Baseline vs Tuned Comparison

```python
lm.compare_baseline_vs_tuned()
```

Shows:

- improvement in R2
- reduction in MSE
- percentage improvement

---

### 13. Best Improvement Model

```python
lm.best_improvement_model()
```

---

## Model Registry

Supported models:

- LinearRegression
- SGDRegressor
- Ridge
- Lasso
- ElasticNet

---

## Notes

- Pipelines are stateless and reusable
- All experiment results are stored internally
- Supports structured reporting
