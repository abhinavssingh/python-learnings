# 📌 MLModelBase

## ✅ Overview

`MLModelBase` is the **core abstract base class** that defines the fundamental contract for all machine learning model wrappers in the framework.

It ensures that all models follow a **consistent lifecycle**:

- Pipeline construction
- Model training
- Prediction

This abstraction enforces **standardization, extensibility, and clean architecture**.

---

## ✅ Architecture Placement

```
MLModelBase ✅ (Abstract Layer)
      ↓
BaseModelWrapper
      ↓
ClassificationModelWrapper / RegressionModelWrapper
```

---

## ✅ Responsibilities

- ✅ Define mandatory methods for all model wrappers
- ✅ Enforce consistent ML workflow
- ✅ Provide a contract for extensibility
- ❌ Does NOT implement any logic
- ❌ Does NOT depend on specific ML libraries

---

## ✅ Method Contracts

### 🔷 build_pipeline(preprocessor)

Constructs the machine learning pipeline.

- Input: Preprocessor (any compatible transformer)
- Output: None (pipeline assigned internally)

---

### 🔷 train(X, y)

Trains the model using the prepared pipeline.

- Input: Features (X), Target (y)
- Output: None

---

### 🔷 predict(X)

Generates predictions from the trained model.

- Input: Features (X)
- Output: Predictions

---

## ✅ Design Principles

| Principle     | Description                                        |
| ------------- | -------------------------------------------------- |
| Abstraction   | Defines contract without implementation            |
| Consistency   | Enforces same method structure across models       |
| Extensibility | Enables custom wrappers easily                     |
| Decoupling    | Independent of frameworks (sklearn, xgboost, etc.) |

---

## ✅ Why This Layer is Important

Without `MLModelBase`:

❌ Different model wrappers may behave inconsistently
❌ Hard to plug into utilities or pipelines
❌ Difficult to scale framework

With `MLModelBase`:

✅ Standardized lifecycle
✅ Plug-and-play wrappers
✅ Clean architecture

---

## ✅ Example Usage

```python
class CustomWrapper(MLModelBase):

    def build_pipeline(self, preprocessor):
        ...

    def train(self, X, y):
        ...

    def predict(self, X):
        ...
```

---

## ✅ Extension Hierarchy

```
MLModelBase
    ↓
BaseModelWrapper
    ↓
Task-specific wrappers:
        - ClassificationModelWrapper
        - RegressionModelWrapper
```

---

## ✅ Best Practices

- ✅ Always implement all abstract methods
- ✅ Keep implementations minimal and focused
- ✅ Delegate logic to lower layers (wrappers, pipelines)
- ✅ Avoid placing business logic here

---

## ✅ Summary

`MLModelBase` provides:

- ✅ Strong architectural foundation
- ✅ Consistent model lifecycle
- ✅ Extensible abstraction layer
- ✅ Clean separation of concerns

---
