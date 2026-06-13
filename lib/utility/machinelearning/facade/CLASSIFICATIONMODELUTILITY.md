
# ClassificationModelUtility – Detailed Documentation

## Overview
The `ClassificationModelUtility` is a high-level orchestration layer designed to manage end-to-end classification workflows. It integrates preprocessing, model execution, evaluation, tuning, and result management into a unified, extensible framework.

---

## Architecture

```
Data → Preprocessing → Model → Metrics → Artifacts → Formatter → UI
```

### Key Responsibilities
- Data preparation and splitting
- Model execution
- Metric computation (via Metrics)
- Artifact extraction
- Hyperparameter tuning
- Result aggregation and ranking

---

## Initialization

```python
cm = ClassificationModelUtility(df, target_col, imputer, outlier_handler)
```

### Parameters
- **df**: Input dataset
- **target_col**: Target column or list (multilabel)
- **imputer**: Optional missing value handler
- **outlier_handler**: Optional outlier processor

---

## Data Preparation

```python
cm.prepare_data()
```

### Features
- Auto-detects problem type
- Supports multilabel via `iterative_train_test_split`
- Applies preprocessing pipeline

### Supported Types
| Type | Strategy |
|------|--------|
| Binary | random split |
| Multiclass | random split |
| Multilabel | iterative split |

---

## Running Experiments

```python
cm.run_experiment("LogisticRegression")
```

### Workflow
1. Fetch model from registry
2. Wrap with OneVsRest (multilabel)
3. Build pipeline
4. Train model
5. Predict outputs
6. Compute probabilities
7. Generate metrics
8. Extract artifacts
9. Store results

---

## Running All Models

```python
cm.run_all_models()
```

### Output
Returns DataFrame of model performance

---

## Artifact Handling

Artifacts include:
- ROC curves
- PR curves
- Confusion matrix (non-multilabel)
- Classification report

Artifacts are separated from scalar metrics:

```
result = {
    metrics,
    artifacts
}
```

---

## Probability Handling

Multilabel probabilities are normalized:

```python
np.column_stack([p[:, 1] for p in raw_proba])
```

---

## Hyperparameter Tuning

```python
cm.tune_model("RandomForest", max_depth=[5,10])
```

Supports:
- Grid search
- Random search

---

## Result Utilities

### Get Results DataFrame

```python
cm.get_results_df()
```

### Get Artifacts Summary

```python
cm.get_artifacts_df()
```

---

## Model Comparison

### Rank Models

```python
cm.rank_models(metric="f1")
```

### Best Model

```python
cm.get_best_model(metric="roc_auc")
```

---

## Confusion Matrix Handling

```python
cm.get_confusion_matrix_df(model_name)
```

Uses `ClassificationFormatter` to convert to DataFrame.

---

## Internal Helpers

### 1. _get_probabilities
Handles model-specific probability output.

### 2. _extract_artifacts
Separates artifacts from metrics to avoid UI pollution.

---

## Design Principles

- ✅ Separation of concerns
- ✅ Modular architecture
- ✅ Extensible model registry
- ✅ Multilabel support
- ✅ Production-ready structure

---

## Best Practices

- Avoid confusion matrix for multilabel
- Always validate label distribution
- Use formatter for UI rendering
- Keep Metrics computation pure

---

## Extensibility

Future capabilities:
- Per-label evaluation
- AutoML integration
- Model explainability
- Feature importance tracking

---

## Summary

`ClassificationModelUtility` acts as the central engine of the ML pipeline, coordinating all stages from data preparation to evaluation and reporting.

