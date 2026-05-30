# HyperparameterTuner Documentation

This document describes the `HyperparameterTuner` utility class used for performing hyperparameter optimization using GridSearchCV and RandomizedSearchCV.

---

## Overview

`HyperparameterTuner` is a lightweight, reusable utility designed to:

- Perform Grid Search and Random Search
- Keep tuning logic separate from model training (clean architecture ✅)
- Return structured results
- Optionally evaluate tuned models on a test dataset

---

## Imports

```python
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
```

---

## Initialization

```python
tuner = HyperparameterTuner(X_train, y_train, X_test, y_test)
```

### Parameters

- `X_train`: Training feature data
- `y_train`: Training target data
- `X_test`: (Optional) Test feature data
- `y_test`: (Optional) Test target data

---

## Grid Search

```python
tuner.grid_search(pipeline, param_grid)
```

### Parameters

- `pipeline`: Scikit-learn pipeline object
- `param_grid`: Dictionary of parameters to search
- `cv`: Number of folds (default = 5)
- `scoring`: Metric to evaluate (default = 'r2')
- `n_jobs`: Parallel processing (default = -1)

### Returns

Dictionary containing:

- `mode`: gridsearch
- `best_params`
- `best_score_cv`
- `cv_results`
- `test_metrics` (if test data provided)

---

## Random Search

```python
tuner.random_search(pipeline, param_distributions)
```

### Parameters

- `pipeline`: Scikit-learn pipeline
- `param_distributions`: Parameter distributions
- `cv`: Number of folds (default = 5)
- `n_iter`: Number of iterations (default = 20)
- `scoring`: Evaluation metric
- `n_jobs`: Parallel jobs

### Returns

Dictionary with:

- `mode`: random_search
- `best_params`
- `best_score_cv`
- `cv_results`
- `test_metrics` (optional)

---

## Error Handling

Raises `ValueError` if:

- `X_train` or `y_train` is missing

---

## Evaluation Metrics

- R2 Score → Higher is better
- Mean Squared Error (MSE) → Lower is better

---

## Example Usage

```python
param_grid = {
    "model__alpha": [0.1, 1.0, 10.0]
}

tuner.grid_search(pipeline, param_grid)
```

---

## Design Benefits

- ✅ Decoupled from model logic
- ✅ Reusable across pipelines
- ✅ Supports both Grid & Random Search
- ✅ Clean output format for reporting
