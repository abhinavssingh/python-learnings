# LinearModelUtility Documentation

## 📌 Overview

`LinearModelUtility` is a **Facade Layer** that orchestrates the end-to-end ML workflow using a **loosely coupled architecture**.

It simplifies:

- Data preparation
- Model training
- Experiment execution
- Hyperparameter tuning
- Model comparison

---

## 🧱 Architecture Role

```
User Code
   ↓
LinearModelUtility (Facade)
   ↓
--------------------------------
| ModelRegistry               |
| Preprocessor                |
| ExperimentRunner            |
| HyperparameterTuner         |
| ModelComparator             |
--------------------------------
```

---

## 🚀 Key Features

✅ Loose coupling (modular design)
✅ Supports multiple models dynamically
✅ Built-in train-test & K-fold validation
✅ Hyperparameter tuning (Grid & Random)
✅ Extensible (plug-and-play models)

---

## 🔧 Constructor

```python
LinearModelUtility(df, target_col, imputer=None, outlier_handler=None)
```

### Parameters

- `df`: Input dataset
- `target_col`: Target column name
- `imputer`: Optional imputer
- `outlier_handler`: Optional outlier handler

---

## 📊 Data Preparation

### `prepare_data()`

Splits data into training and test sets and builds preprocessing pipeline.

```python
lm.prepare_data(test_size=0.2)
```

---

## ⚙️ Model Training

### `run_experiment()`

Runs a single model experiment.

**Modes:**

- Train-Test
- K-Fold Cross Validation

```python
lm.run_experiment("Ridge", k_fold=5)
```

---

## 🔁 Run Multiple Models

### `run_all_models()`

Runs all registered models.

```python
lm.run_all_models()
```

---

### `run_experiments()`

Executes multiple experiment configurations.

```python
configs = [
    {"model_name": "Ridge", "k_fold": 5},
    {"model_name": "Lasso"}
]

lm.run_experiments(configs)
```

---

## 🔍 Hyperparameter Tuning

### `grid_search_cv()`

Performs Grid Search.

```python
lm.grid_search_cv("Ridge", param_grid)
```

---

### `tune_model()`

Supports both Grid and Random Search.

```python
lm.tune_model("Ridge", param_grid, search_type="random")
```

---

## 📈 Results Handling

### `get_results_df()`

Returns experiment results as DataFrame.

---

### `rank_models()`

Ranks models based on metric.

---

### `get_best_model()`

Returns best-performing model.

---

### `compare_models()`

Aggregates model performance.

---

## 🔄 Experiment Output Structure

Example result row:

```
{
  "model": "Ridge",
  "mode": "train-test",
  "type": "baseline",
  "R2": 0.82,
  "MSE": 120
}
```

### ✅ For Tuned Models

```
{
  "model": "Ridge",
  "type": "tuned",
  "mode": "gridsearch",
  "param_alpha": 10.0,
  "R2": 0.85
}
```

---

## 🧠 Design Principles

### ✅ Facade Pattern

Simplifies complex ML workflows.

### ✅ Single Responsibility

Each component handles one task:

- Registry → model selection
- Runner → execution
- Tuner → optimization

### ✅ Loose Coupling

Components can be replaced independently.

---

## ⚠️ Best Practices

- Always flatten `best_params` into columns
- Avoid storing nested dictionaries for analysis
- Use ModelComparator for ranking

---

## 🚀 Example Usage

```python
lm = LinearModelUtility(df, "target")

lm.prepare_data()

lm.run_all_models()

lm.grid_search_cv("Ridge", param_grid)

best_model = lm.get_best_model()
```

---

## 🏁 Summary

`LinearModelUtility` acts as a **central engine** for:

✅ Running experiments
✅ Managing models
✅ Performing tuning
✅ Producing structured results

---

## 🔮 Future Enhancements

- AutoML integration
- MLflow experiment tracking
- Parallel model training
- Dashboard integration

---
