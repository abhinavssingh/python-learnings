# 📌 ClassificationModelWrapper

## ✅ Overview

`ClassificationModelWrapper` is a **task-specific wrapper** responsible for building and executing classification pipelines.

It extends `BaseModelWrapper` and introduces:

- Pipeline construction
- Imbalance handling (SMOTE integration)
- Probability prediction support
- Classification metrics evaluation

---

## ✅ Architecture Placement

```
BaseModelWrapper
     ↓
ClassificationModelWrapper ✅
     ↓
Pipeline:
    Preprocessor
    ↓
    ImbalanceHandler (SMOTE)
    ↓
    Model
```

---

## ✅ Responsibilities

- ✅ Construct ML pipeline
- ✅ Flatten nested preprocessors
- ✅ Inject imbalance handler into pipeline
- ✅ Support probability predictions
- ✅ Evaluate classification metrics
- ❌ Does NOT handle data preparation
- ❌ Does NOT manage model registry

---

## ✅ Key Features

| Feature                | Supported |
| ---------------------- | --------- |
| Pipeline construction  | ✅        |
| Recursive flattening   | ✅        |
| SMOTE integration      | ✅        |
| CV-safe execution      | ✅        |
| Probability prediction | ✅        |
| Metrics evaluation     | ✅        |

---

## ✅ Pipeline Composition

```
Preprocessor (flattened)
    ↓
SMOTE (optional)
    ↓
Model
```

---

## ✅ Important Design Decisions

### ✅ 1. Recursive Flattening

Avoids nested pipelines:

❌ Invalid:

```
Pipeline([
    ("preprocessor", Pipeline(...))
])
```

✅ Correct:

```
Step1 → Step2 → Step3
```

---

### ✅ 2. SMOTE Placement

SMOTE is applied **after preprocessing and before model training**:

```
Preprocessor → SMOTE → Model
```

---

### ✅ 3. Safe Probability Prediction

Handles models that may not support `predict_proba()`:

```python
if hasattr(self.pipeline, "predict_proba"):
```

---

## ✅ Execution Flow

```
build_pipeline()
    ↓
train()
    ↓
predict()
    ↓
evaluate()
```

---

## ✅ Best Practices

- ✅ Always flatten preprocessors
- ✅ Inject imbalance handler before building pipeline
- ✅ Use `imblearn.pipeline.Pipeline` for SMOTE support
- ✅ Keep wrapper focused on pipeline logic only

---

## ✅ Summary

`ClassificationModelWrapper` provides:

- ✅ Clean pipeline construction
- ✅ SMOTE integration
- ✅ Safe prediction handling
- ✅ Evaluation support
- ✅ Extensible architecture

---
