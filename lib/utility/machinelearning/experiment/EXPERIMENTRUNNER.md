
# ExperimentRunner – Detailed Documentation

## Overview

The `ExperimentRunner` class is responsible for executing machine learning experiments in a consistent and reusable manner. It handles model training, evaluation, and result collection.

---

## Purpose

- Provide a clean abstraction for running experiments
- Standardize model execution flow
- Store and manage experiment results
- Serve as a reusable engine for utilities like `ClassificationModelUtility`

---

## Class Structure

```python
class ExperimentRunner:
```

---

## Initialization

```python
ExperimentRunner(preprocessor)
```

### Parameters

- **preprocessor**: Preprocessing pipeline applied before model training

### Attributes

- `self.preprocessor` → Holds preprocessing pipeline
- `self.results` → Stores baseline experiment results
- `self.tuned_results` → Stores tuned results (future use / extension)

---

## Method: run()

```python
run(model_name, wrapper, X_train, X_test, y_train, y_test)
```

### Workflow

1. Build pipeline using preprocessor
2. Train model
3. Generate predictions
4. Evaluate model
5. Store results

---

### Step-by-Step Execution

```python
wrapper.build_pipeline(self.preprocessor)
```
- Attaches preprocessing + model into single pipeline

```python
wrapper.train(X_train, y_train)
```
- Trains model

```python
preds = wrapper.predict(X_test)
```
- Generates predictions

```python
metrics = wrapper.evaluate(y_test, preds)
```
- Computes evaluation metrics

```python
result = {
    "model": model_name,
    **metrics
}
```
- Combines model name with metrics

---

### Output

```python
{
    "model": str,
    "accuracy": float,
    "f1": float,
    ...
}
```

---

## Method: run_all()

```python
run_all(model_registry, X_train, X_test, y_train, y_test)
```

### Parameters

- **model_registry**: Dictionary of model wrappers
- **X_train, X_test, y_train, y_test**: Dataset splits

---

### Workflow

```python
for name, wrapper in model_registry.items():
    self.run(name, wrapper, ...)
```

- Iterates through all models
- Executes each experiment
- Stores results

---

### Returns

```python
list[dict]
```

Example:

```python
[
    {"model": "LogisticRegression", "accuracy": 0.85},
    {"model": "RandomForest", "accuracy": 0.88}
]
```

---

## Design Principles

- ✅ Single responsibility (experiment execution)
- ✅ Decoupled from model definitions
- ✅ Reusable across classification and regression
- ✅ Clean orchestration layer

---

## Integration

Used by:

- `ClassificationModelUtility`
- `LinearModelUtility`

---

## Limitations (Current)

- No tuning integration directly
- No artifact handling
- No experiment metadata tracking

---

## Recommended Enhancements

- Add experiment metadata (timestamp, params)
- Integrate artifact tracking (ROC, confusion matrix)
- Support parallel execution
- Add logging support

---

## Summary

`ExperimentRunner` acts as a lightweight execution engine for ML experiments. It ensures consistent training and evaluation across models and simplifies orchestration within higher-level utilities.

