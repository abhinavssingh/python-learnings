# HyperparameterTuner - Comprehensive Documentation

## 📌 Overview

The `HyperparameterTuner` is a **production-grade hyperparameter optimization engine** designed for modern ML systems.

It supports:

- ✅ Grid Search (exhaustive search)
- ✅ Random Search (probabilistic search)
- ✅ Experiment-level tracking
- ✅ AutoML-ready integration
- ✅ Visualization-ready outputs (flattened, row-based)

---

# 🚀 Key Evolution

### ❌ Before

- Returned single result (best params only)
- Nested `cv_results` (hard to use)
- Not suitable for visualization

### ✅ Now

- Returns **multiple rows** (each param combination)
- Flattened `param_*` columns ✅
- Compatible with visualization modules ✅
- Integrated experiment naming ✅

---

# 🧱 Architecture Role

```
LinearModelUtility
        ↓
HyperparameterTuner
        ↓
DataCleaner
        ↓
Visualization Layer
```

---

## 🧠 Core Responsibilities

- Perform tuning using sklearn
- Convert `cv_results_` → clean tabular format
- Attach experiment metadata
- Provide test evaluation

---

## ⚙️ Initialization

```python
tuner = HyperparameterTuner(X_train, y_train, X_test, y_test)
```

### Parameters

- `X_train`: Training features
- `y_train`: Training target
- `X_test`: Optional test features
- `y_test`: Optional test target

---

## 🔍 Grid Search

```python
tuner.grid_search(pipeline, model_name, param_grid)
```

### ✅ Returns

List of dictionaries:

Each row = one hyperparameter configuration

| Field      | Description      |
| ---------- | ---------------- |
| model      | model name       |
| experiment | experiment label |
| mode       | gridsearch       |
| type       | tuned            |
| score      | mean CV score    |
| param\_\*  | hyperparameters  |

---

## 🎲 Random Search

```python
tuner.random_search(pipeline, model_name, param_distributions)
```

### ✅ Returns

Same structure as grid search

---

## 🔄 Data Transformation (CRITICAL)

Internally uses:

```python
DataCleaner.flatten_cv_results()
```

### ✅ Converts:

```
cv_results_ (dict) ❌
```

Into:

```
Tabular rows ✅
```

---

## 🧪 Evaluation Metrics

- ✅ `score` → cross-validation metric
- ✅ `R2` / `MSE` → test metrics (optional)

---

## 📊 Output Example

```
model | param_alpha | score | mode
-----------------------------------
Ridge | 0.1         | 0.98  | gridsearch
Ridge | 1.0         | 0.99  | gridsearch
```

---

## ✅ Design Principles

### ✅ 1. Separation of Concerns

- Tuning logic isolated from model execution

### ✅ 2. Visualization-Ready Output

- No nested structures
- Flat schema

### ✅ 3. Experiment-Aware

- Uses `ExperimentNameBuilder`

### ✅ 4. Scalable Design

- Works with AutoML pipelines

---

## ⚠️ Error Handling

Raises:

```python
ValueError
```

If training data is missing.

---

## ✅ Example Usage

```python
param_grid = {
    "model__alpha": [0.1, 1, 10]
}

results = tuner.grid_search(pipeline, "Ridge", param_grid)
```

---

## 🚀 Integration Highlights

Works seamlessly with:

- ✅ LinearModelUtility
- ✅ HyperparameterPlots
- ✅ OptimizationPlots
- ✅ ComparisonPlots

---

## 🏁 Conclusion

The updated `HyperparameterTuner` is a **core building block of an AutoML-ready system**, enabling:

✔ Full hyperparameter exploration
✔ Visualization of tuning landscapes
✔ Experiment-level tracking
✔ Scalable ML workflows

---
