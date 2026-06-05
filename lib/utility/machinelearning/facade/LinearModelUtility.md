# LinearModelUtility - Comprehensive Documentation

## 📌 Overview

`LinearModelUtility` is a **production-grade Facade Layer** that orchestrates the end-to-end ML lifecycle using a **loosely coupled and experiment-driven architecture**.

It acts as the central engine for:

- ✅ Data preparation
- ✅ Model training
- ✅ Experiment execution
- ✅ Hyperparameter tuning
- ✅ Experiment tracking (NEW)
- ✅ Model comparison & ranking

---

# 🚀 Key Evolution (IMPORTANT)

### ❌ Earlier Version

- Model-level execution
- Limited tracking (only baseline vs tuned)
- Best params only (no full tuning visibility)

### ✅ Current Version

- ✅ Experiment-level tracking (`experiment` column)
- ✅ Hyperparameter-aware outputs (`param_*` columns)
- ✅ Multi-row tuning results
- ✅ Visualization-ready architecture
- ✅ AutoML-ready framework

---

# 🧱 Architecture Role

```
User Code
   ↓
LinearModelUtility (Facade)
   ↓
-----------------------------------------
| ModelRegistry                         |
| Preprocessor                          |
| ExperimentRunner                      |
| HyperparameterTuner ✅               |
| ModelComparator ✅                    |
| Formatter ✅              |
-----------------------------------------
```

---

# 🧠 Core Responsibilities

- Coordinate all ML operations from a single interface
- Maintain experiment-level consistency across models
- Manage structured results for downstream visualization
- Enable plug-and-play ML components

---

# 🚀 Key Features

✅ Loose coupling (modular, replaceable components)
✅ Dynamic model registry
✅ Built-in Train-Test & K-Fold validation
✅ Hyperparameter tuning (Grid + Random)
✅ Experiment-level tracking (NEW)
✅ Multi-model experimentation
✅ Visualization-ready output

---

# 🔧 Constructor

```python
LinearModelUtility(df, target_col, imputer=None, outlier_handler=None)
```

### Parameters

- `df`: Input dataset
- `target_col`: Target column
- `imputer`: Optional preprocessing
- `outlier_handler`: Optional preprocessing

---

# 📊 Data Preparation

### `prepare_data()`

Splits data and builds preprocessing pipeline.

```python
lm.prepare_data(test_size=0.2)
```

---

# ⚙️ Model Training

## ✅ `run_experiment()`

Runs a single experiment.

### ✅ Supported Modes

- Train-Test
- K-Fold Cross Validation

### ✅ NEW: Experiment Tracking

Each run produces:

```
experiment = "Ridge | kfold=5 | imputer=SimpleImputer"
```

---

# 🔁 Batch Execution

## ✅ `run_all_models()`

Runs all registered models.

## ✅ `run_experiments()`

Executes multiple configurations.

```python
configs = [
    {"model_name": "Ridge", "k_fold": 5},
    {"model_name": "Lasso"}
]
```

---

# 🔍 Hyperparameter Tuning

## ✅ `grid_search_cv()`

Performs traditional grid search.

## ✅ `tune_model()`

Advanced tuning (Grid / Random).

### ✅ NEW Behavior

- Returns **multiple rows** (each param combination)
- Uses `HyperparameterTuner`
- Adds experiment metadata

---

# 📈 Results Structure

## ✅ Baseline Example

```
{
  "model": "Ridge",
  "experiment": "Ridge | train-test",
  "mode": "train-test",
  "type": "baseline",
  "R2": 0.98
}
```

---

## ✅ Tuned Example

```
{
  "model": "ElasticNet",
  "experiment": "ElasticNet | random",
  "mode": "random_search",
  "type": "tuned",
  "param_model__alpha": 0.1,
  "param_model__l1_ratio": 0.5,
  "score": 0.97
}
```

---

# 📊 Results Utilities

## ✅ `get_results_df()`

Returns complete experiment dataset.

## ✅ `rank_models()`

Ranks based on metric (supports experiment-level ranking).

## ✅ `get_best_model()`

Returns best-performing experiment.

## ✅ `compare_models()`

Aggregates performance results.

---

# ✅ Experiment Schema (IMPORTANT)

Every row follows:

| Column        | Purpose           |
| ------------- | ----------------- |
| model         | model name        |
| experiment ✅ | unique identifier |
| type          | baseline / tuned  |
| mode          | execution type    |
| metrics       | R2 / MSE / score  |
| param\_\* ✅  | hyperparameters   |

---

# 🧠 Design Principles

## ✅ Facade Pattern

Simplifies multi-step ML workflows.

## ✅ Separation of Concerns

- Registry → model selection
- Runner → execution
- Tuner → optimization
- Cleaner → data safety

## ✅ Experiment-Centric Design (NEW)

Moves from:

```
model comparison ❌
```

to:

```
experiment comparison ✅
```

---

# ⚠️ Best Practices

- Always use `experiment` for plotting
- Flatten hyperparameters (`param_*`)
- Avoid nested dictionaries (`cv_results`)
- Use `score` for tuning results

---

# 🚀 Example Usage

```python
lm = LinearModelUtility(df, "target")

lm.prepare_data()

lm.run_all_models()

lm.tune_model("Ridge", param_grid)

results_df = lm.get_results_df()
```

---

# 🔗 Integration with Visualization

Works seamlessly with:

- ✅ ModelPerformanceVisualizer
- ✅ ComparisonPlots
- ✅ HyperparameterPlots
- ✅ OptimizationPlots

---

# 🚀 Advanced Capabilities

✅ Multi-experiment comparison
✅ Hyperparameter exploration
✅ AutoML-ready workflows
✅ Experiment tracking foundation

---

# 🔮 Future Enhancements

- MLflow-style experiment tracking
- Experiment IDs + timestamps
- AutoML pipeline automation
- Interactive dashboards (Streamlit)

---

# 🏁 Conclusion

`LinearModelUtility` is the **central orchestration engine** powering:

✔ Experiment-driven ML pipelines
✔ Hyperparameter tuning workflows
✔ Visualization-ready data outputs
✔ Scalable and modular ML architecture

---
