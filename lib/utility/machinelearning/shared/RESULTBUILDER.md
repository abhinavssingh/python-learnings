# 📌 ResultBuilder

## ✅ Overview

`ResultBuilder` is a centralized utility class responsible for constructing **standardized result objects** across all machine learning workflows within the framework.

It ensures consistency across:

- Baseline experiments
- Ensemble models
- Hyperparameter tuning
- Future AutoML pipelines

This class enforces a **unified schema** that enables seamless comparison, visualization, and reporting.

---

## ✅ Responsibilities

- ✅ Standardize result structure across all workflows
- ✅ Merge metrics, metadata, and artifacts into a single object
- ✅ Handle imbalance-related flags and summaries
- ✅ Support extensibility via dynamic fields (`**metrics`, `extra`)
- ✅ Ensure compatibility with downstream systems (DataFrame, reporting, visualization)

---

## ✅ Key Design Concepts

### 🔷 Unified Result Schema

All experiment types (baseline, ensemble, tuned) follow the same structure:

```
model → metadata → metrics → artifacts
```

---

### 🔷 Dynamic Metrics Injection

Supports flexible metric addition using `**metrics`:

```python
accuracy=0.9,
f1=0.88,
precision=0.91
```

---

### 🔷 Imbalance Awareness

Automatically tracks:

- Whether imbalance handling was applied
- Which method was used (e.g., SMOTE)
- Before/after distribution (stored in artifacts)

---

### 🔷 Extensibility via `extra`

Supports workflow-specific metadata:

```python
extra={
    "ensemble_type": "voting",
    "method": "soft",
    "base_models": [...]
}
```

---

## ✅ Output Structure

```python
{
  "model": "LogisticRegression",
  "family": "linear",
  "experiment": "LogisticRegression | classification",
  "mode": "train-test",
  "type": "baseline",

  "imbalance_applied": True,
  "imbalance_method": "SMOTE",

  "accuracy": 0.90,
  "f1": 0.88,

  "artifacts": {
      "roc_curve": ...,
      "confusion_matrix": ...,
      "imbalance": {
          "before": {...},
          "after": {...}
      }
  }
}
```

---

## ✅ Usage Examples

### 🔷 Baseline Model

```python
result = ResultBuilder.build(
    model="LogisticRegression",
    family="linear",
    result_type="baseline",
    imbalance_summary=imbalance_summary,
    artifacts=artifacts,
    accuracy=0.9,
    f1=0.88
)
```

---

### 🔷 Ensemble Model

```python
result = ResultBuilder.build(
    model="Ensemble_Voting",
    family="ensemble",
    result_type="ensemble",
    imbalance_summary=imbalance_summary,
    artifacts=artifacts,
    extra={
        "ensemble_type": "parallel",
        "method": "voting"
    },
    accuracy=0.92,
    f1=0.90
)
```

---

### 🔷 Tuned Model

```python
result = ResultBuilder.build(
    model="RandomForest",
    result_type="tuned",
    mode="cv",
    artifacts=artifacts,
    extra={
        "search_type": "grid",
        "best_params": {...}
    },
    accuracy=0.93
)
```

---

## ✅ Best Practices

- ✅ Always use `ResultBuilder` instead of manual result construction
- ✅ Keep metrics flattened (no nested structures)
- ✅ Store complex outputs in `artifacts`
- ✅ Use `extra` for workflow-specific metadata

---

## ✅ Benefits

- ✅ Consistent schema across system
- ✅ Simplified DataFrame conversion
- ✅ Easier model comparison and ranking
- ✅ Cleaner orchestration layer
- ✅ AutoML-ready design

---

## ✅ Future Enhancements

- ✅ Add timestamp and execution time
- ✅ Add schema validation layer
- ✅ Add versioning support
- ✅ Integration with reporting dashboards

---

## ✅ Summary

`ResultBuilder` provides a **standardized, extensible, and production-grade mechanism** for constructing experiment results, enabling consistency, scalability, and maintainability across the entire ML framework.

---
