# 📌 SequentialEnsembleWrapper

## ✅ Overview

`SequentialEnsembleWrapper` is an ensemble wrapper designed to support **sequential learning techniques (boosting)** within the machine learning framework.

It extends `EnsembleModelWrapper` and integrates seamlessly with:

- Wrapper-based architecture
- Preprocessor pipeline
- Imbalance handling (SMOTE)
- Classification evaluation layer

Sequential ensembles differ from parallel ensembles by training models in a **dependent sequence**, where each model improves upon the errors of the previous model.

---

## ✅ Responsibilities

- ✅ Build boosting-based ensemble models
- ✅ Support:
  - AdaBoost
  - Gradient Boosting
  - Custom boosting models
- ✅ Integrate seamlessly into pipeline execution
- ✅ Support SMOTE-based imbalance handling
- ✅ Maintain compatibility with classification metrics and artifacts

---

## ✅ Architecture Placement

```
ClassificationModelWrapper
        ↓
EnsembleModelWrapper
        ↓
SequentialEnsembleWrapper ✅
```

---

## ✅ Supported Methods

| Method            | Description                           |
| ----------------- | ------------------------------------- |
| adaboost          | Adaptive boosting using weak learners |
| gradient_boosting | Gradient-based sequential boosting    |

---

## ✅ Custom Model Support

You can pass any compatible boosting model:

```python
from sklearn.ensemble import GradientBoostingClassifier

config = {
    "type": "sequential",
    "model": GradientBoostingClassifier(n_estimators=200)
}
```

---

## ✅ Pipeline Flow

```
Preprocessor
    ↓
SMOTE (optional)
    ↓
Boosting Model (Sequential Learning)
```

---

## ✅ Usage Example

### ✅ Method-based

```python
config = {
    "type": "sequential",
    "method": "gradient_boosting"
}

cm.run_ensemble(config)
```

### ✅ Custom Model

```python
from sklearn.ensemble import AdaBoostClassifier

config = {
    "type": "sequential",
    "model": AdaBoostClassifier(n_estimators=100)
}

cm.run_ensemble(config)
```

---

## ✅ Key Design Features

- ✅ Explicit handling of sklearn models (no truthiness errors)
- ✅ Supports both predefined and custom boosting models
- ✅ SMOTE-compatible pipeline integration
- ✅ Config-driven execution
- ✅ Clean inheritance from classification wrapper hierarchy

---

## ✅ Important Notes

### ⚠️ Avoid Boolean Checks on Models

Always use:

```python
if model is not None
```

Do NOT use:

```python
if model  # ❌ Causes sklearn AttributeError
```

---

## ✅ Best Practices

- ✅ Prefer Gradient Boosting for structured/tabular data
- ✅ Use SMOTE for imbalanced datasets
- ✅ Tune boosting parameters for optimal performance
- ✅ Use custom models for advanced control

---

## ✅ Limitations

- ❌ Does not support wrapper chaining (not required for boosting)
- ❌ Limited to boosting-style algorithms
- ❌ Does not expose intermediate learners

---

## ✅ Future Extensions

- ✅ XGBoost / LightGBM integration
- ✅ Custom boosting strategies
- ✅ AutoML-based selection (boosting vs voting vs stacking)
- ✅ Early stopping + evaluation tracking

---

## ✅ Summary

`SequentialEnsembleWrapper` provides:

- ✅ Robust boosting-based ensemble support
- ✅ Clean pipeline and SMOTE integration
- ✅ Flexible configuration (method or custom model)
- ✅ Stable, production-ready behavior

---
