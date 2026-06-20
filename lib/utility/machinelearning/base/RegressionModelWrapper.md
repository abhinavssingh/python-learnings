# 📌 RegressionModelWrapper

## ✅ Overview

`RegressionModelWrapper` is a **task-specific wrapper** designed for regression problems.
It extends `BaseModelWrapper` and is responsible for:

- Constructing a regression pipeline
- Executing regression models
- Evaluating regression metrics

Unlike classification, regression does **not require probability prediction**, making the implementation simpler and specialized.

---

## ✅ Architecture Placement

```
BaseModelWrapper
     ↓
RegressionModelWrapper ✅
     ↓
Pipeline:
    Preprocessor
    ↓
    Model
```

---

## ✅ Responsibilities

- ✅ Build pipeline for regression models
- ✅ Execute model training and prediction
- ✅ Evaluate regression performance
- ✅ Maintain separation from classification logic
- ❌ Does NOT support probability predictions
- ❌ Does NOT include imbalance handling

---

## ✅ Key Features

| Feature                   | Supported |
| ------------------------- | --------- |
| Pipeline construction     | ✅        |
| Regression evaluation     | ✅        |
| Preprocessing integration | ✅        |
| Probability prediction    | ❌        |
| Imbalance handling        | ❌        |

---

## ✅ Pipeline Composition

```
Preprocessor
    ↓
Model
```

---

## ✅ Design Decisions

### ✅ 1. Simplicity Over Complexity

Regression problems do not require:

- Probability outputs
- Class balancing

This keeps the wrapper clean and focused.

---

### ✅ 2. Separation from Classification

Classification-specific features such as:

- SMOTE
- Precision/Recall/F1

are NOT included here.

---

### ✅ 3. Standard Pipeline Usage

Uses `sklearn.pipeline.Pipeline` because:

- No resampling is required
- Simpler execution flow

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

## ✅ Evaluation Metrics

Handled via:

```python
Metrics.regression()
```

Typical metrics may include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

---

## ✅ Example Usage

```python
wrapper = RegressionModelWrapper(model)

wrapper.build_pipeline(preprocessor)
wrapper.train(X_train, y_train)

y_pred = wrapper.predict(X_test)
metrics = wrapper.evaluate(y_test, y_pred)
```

---

## ✅ Best Practices

- ✅ Always build pipeline before training
- ✅ Keep preprocessing consistent with training data
- ✅ Use appropriate regression metrics
- ✅ Avoid mixing classification logic

---

## ✅ Extension Possibilities

Future enhancements can include:

- Residual analysis visualization
- Error distribution plots
- Prediction intervals
- Model explainability (SHAP)

---

## ✅ Summary

`RegressionModelWrapper` provides:

- ✅ Clean pipeline abstraction for regression
- ✅ Proper evaluation flow
- ✅ Clear separation from classification concerns
- ✅ Extensible structure for future enhancements

---
