# 📌 StackingEnsembleWrapper

## ✅ Overview

`StackingEnsembleWrapper` is an advanced ensemble wrapper designed to implement **stacked generalization (stacking)** within the machine learning framework.

Stacking combines multiple base models (level-0 learners) and uses a **meta-learner (final estimator)** to learn optimal combinations of their predictions.

This wrapper extends `EnsembleModelWrapper` and integrates seamlessly with:

- Wrapper-based architecture
- Preprocessing pipeline
- Imbalance handling (SMOTE)
- Evaluation and artifact extraction

---

## ✅ Responsibilities

- ✅ Combine predictions of multiple base models
- ✅ Train a meta-learner on top of base model outputs
- ✅ Support configurable meta-models
- ✅ Integrate into pipeline execution flow
- ✅ Maintain compatibility with classification metrics and artifacts

---

## ✅ Architecture Placement

```
ClassificationModelWrapper
        ↓
EnsembleModelWrapper
        ↓
StackingEnsembleWrapper ✅
```

---

## ✅ Key Concepts

### 🔷 Base Models (Level-0 Learners)

- Models trained independently
- Their predictions become inputs to the meta-model

### 🔷 Meta Model (Level-1 Learner)

- Learns how to combine base model predictions
- Improves overall predictive performance

---

## ✅ Pipeline Flow

```
Preprocessor
    ↓
SMOTE (optional)
    ↓
Base Models (parallel)
    ↓
Meta Model (final estimator)
```

---

## ✅ Usage Example

```python
config = {
    "type": "stacking",
    "model_names": [
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBoost"
    ],
    "meta_model": "LogisticRegression"
}

cm.run_ensemble(config)
```

---

## ✅ Meta Model Requirements

The `final_estimator` must:

- ✅ Implement `fit()`
- ✅ Be compatible with sklearn pipelines
- ✅ Typically be a simple model (e.g., LogisticRegression)

---

## ✅ Design Features

- ✅ Hybrid ensemble (parallel + sequential learning)
- ✅ Wrapper-driven architecture
- ✅ SMOTE-compatible
- ✅ Config-driven execution
- ✅ Fully compatible with metrics and artifacts

---

## ✅ Best Practices

- ✅ Use diverse base models (linear + tree + boosting)
- ✅ Choose simple meta-models (e.g., LogisticRegression)
- ✅ Tune base models before stacking
- ✅ Avoid overly complex meta-models (risk of overfitting)

---

## ✅ Limitations

- ❌ Increased training time
- ❌ Higher complexity
- ❌ Requires careful model selection

---

## ✅ Future Enhancements

- ✅ Passthrough mode support (include original features)
- ✅ Auto meta-model selection
- ✅ Model diversity scoring
- ✅ AutoML-based ensemble selection

---

## ✅ Summary

`StackingEnsembleWrapper` provides:

- ✅ Advanced ensemble learning capability
- ✅ Meta-learning for improved performance
- ✅ Seamless integration with pipeline and SMOTE
- ✅ Flexible and extensible design

---
