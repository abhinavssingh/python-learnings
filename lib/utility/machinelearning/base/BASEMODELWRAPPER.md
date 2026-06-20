# 📌 BaseModelWrapper

## ✅ Overview

`BaseModelWrapper` is the **core abstraction layer** responsible for:

- Managing model lifecycle (train, predict)
- Holding the execution pipeline
- Supporting dynamic injection of behaviors such as **imbalance handling (e.g., SMOTE)**

It extends `MLModelBase` and acts as the **foundation for all model wrappers** (classification, regression, etc.).

---

## ✅ Architecture Placement

```
MLModelBase
↓
BaseModelWrapper ✅
↓
ClassificationModelWrapper
↓
Pipeline:
Preprocessor
↓
ImbalanceHandler (optional)
↓
Model
```

---

## ✅ Responsibilities

- ✅ Store model instance
- ✅ Manage pipeline lifecycle
- ✅ Execute training (`fit`)
- ✅ Execute inference (`predict`)
- ✅ Support imbalance handler injection
- ❌ Does NOT build pipeline (handled by child classes)
- ❌ Does NOT handle metrics (handled separately)

---

## ✅ Key Design Principles

| Principle              | Description                                    |
| ---------------------- | ---------------------------------------------- |
| Separation of Concerns | Pipeline building is NOT handled here          |
| Extensibility          | Supports injection of additional behaviors     |
| Reusability            | Common logic shared across all model wrappers  |
| Loose Coupling         | No direct dependency on SMOTE or preprocessing |
| Clean Abstraction      | Keeps execution separate from configuration    |

---

## ✅ Imbalance Handling Integration

🔷 Injection

```Python
wrapper.set_imbalance_handler(SMOTEHandler(...))
```

---

## 🔷 Usage Flow

```Utility Layer
↓
Inject SMOTEHandler ✅
↓
ClassificationModelWrapper builds pipeline
↓
Pipeline executes SMOTE during training
```

---

## ✅ Important

- BaseModelWrapper does NOT apply SMOTE directly
- It only stores the handler
- Pipeline (child layer) decides how to use it

---

## ✅ Methods Summary

| Method                  | Purpose                       |
| ----------------------- | ----------------------------- |
| **init**                | Initialize model and pipeline |
| set_imbalance_handler() | Inject imbalance strategy     |
| get_pipeline()          | Safely retrieve pipeline      |
| train()                 | Train model through pipeline  |
| predict()               | Perform inference             |

---

## ✅ Execution Flow

```python
wrapper = BaseModelWrapper(model)

wrapper.set_pipeline(...) # via child class
wrapper.train(X_train, y_train)
wrapper.predict(X_test)
```

---

## ✅ Error Handling

🔴 Pipeline Not Built

```Python
ValueError: Pipeline not built for model: <ModelName>
```

✅ Ensures:

- No accidental execution
- Enforces correct lifecycle usage

---

## ✅ Extension Points

This class is designed to support:
✅ Child Wrappers

- ClassificationModelWrapper
- RegressionModelWrapper

---

## ✅ Behavioral Extensions

- Imbalance handling (SMOTE, ADASYN)
- Feature selection
- Custom pipeline injections

---

## ✅ What This Class Does NOT Do

| Responsibility        | Reason                 |
| --------------------- | ---------------------- |
| Pipeline construction | Done in child wrappers |
| Metrics evaluation    | Done in Metrics layer  |
| Data preprocessing    | Done in Preprocessor   |
| Hyperparameter tuning | Done in Tuner          |

## ✅ Example Usage

```Python
wrapper = BaseModelWrapper(model)

wrapper.set_imbalance_handler(smote_handler)

wrapper.build_pipeline(preprocessor) # via subclass

wrapper.train(X_train, y_train)

preds = wrapper.predict(X_test)
```

---

## ✅ Best Practices

- ✅ Always call build_pipeline() before train()
- ✅ Inject imbalance handler BEFORE building pipeline
- ✅ Keep BaseModelWrapper lightweight
- ✅ Avoid adding task-specific logic here

---

## ✅ Summary

BaseModelWrapper provides:

- ✅ Central execution control
- ✅ Clean extensibility model
- ✅ Safe pipeline handling
- ✅ Plug-and-play capabilities
- ✅ Strong separation of concerns
