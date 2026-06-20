# 📌 ParallelEnsembleWrapper

## ✅ Overview

`ParallelEnsembleWrapper` is an ensemble model wrapper designed to support **parallel ensemble learning techniques** such as **Voting** and **Bagging** within the machine learning framework.

It extends `EnsembleModelWrapper` and integrates seamlessly with:

- Wrapper-based architecture
- Imbalance handling (SMOTE)
- imblearn pipeline execution

---

## ✅ Responsibilities

- ✅ Combine multiple base models into a single ensemble
- ✅ Support:
  - Voting (soft/hard)
  - Bagging
- ✅ Integrate with preprocessing pipeline
- ✅ Support imbalance handling via SMOTE
- ✅ Maintain compatibility with classification evaluation

---

## ✅ Architecture Placement

```
ClassificationModelWrapper
        ↓
EnsembleModelWrapper
        ↓
ParallelEnsembleWrapper ✅
```

---

## ✅ Supported Methods

| Method  | Description                                   |
| ------- | --------------------------------------------- |
| Voting  | Combines predictions from multiple models     |
| Bagging | Uses bootstrap sampling with a base estimator |

---

## ✅ Voting Modes

| Voting Type | Behavior                               |
| ----------- | -------------------------------------- |
| soft        | Uses class probabilities (recommended) |
| hard        | Uses majority voting                   |

---

## ✅ Pipeline Flow

```
Preprocessor
    ↓
SMOTE (optional)
    ↓
Ensemble Model (Voting / Bagging)
```

---

## ✅ Usage Example

```python
config = {
    "type": "parallel",
    "method": "voting",
    "voting": "soft",
    "model_names": [
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBoost"
    ]
}

cm.run_ensemble(config)
```

---

## ✅ Key Design Features

- ✅ Wrapper-based implementation
- ✅ Config-driven behavior
- ✅ SMOTE compatible
- ✅ Pipeline-safe execution
- ✅ Extensible for future ensemble strategies

---

## ✅ Best Practices

- ✅ Prefer soft voting when probability is available
- ✅ Ensure base models support `predict_proba` for soft voting
- ✅ Keep models diverse for better ensemble performance
- ✅ Use registry-based model resolution for scalability

---

## ✅ Limitations

- ❌ Bagging currently uses only one base estimator
- ❌ Hard voting does not support probability outputs
- ❌ No built-in model weighting (can be extended)

---

## ✅ Future Extensions

- ✅ Weighted voting
- ✅ Dynamic model selection
- ✅ AutoML ensemble builder
- ✅ Diversity-aware ensemble selection

---

## ✅ Summary

`ParallelEnsembleWrapper` provides:

- ✅ Scalable parallel ensemble support
- ✅ Clean integration with pipeline & SMOTE
- ✅ Config-driven flexibility
- ✅ Reusable and extensible design

---
