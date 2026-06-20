# 📌 SMOTEHandler

## ✅ Overview

`SMOTEHandler` is an **imbalance handling component** designed to integrate seamlessly with your machine learning framework using the **imblearn pipeline**.

It applies **Synthetic Minority Oversampling Technique (SMOTE)** to balance class distributions during training.

---

## ✅ Responsibilities

- ✅ Perform **oversampling of minority class**
- ✅ Integrate into `imblearn.pipeline.Pipeline`
- ✅ Capture **class distribution before and after resampling**
- ✅ Provide **audit and explainability support**
- ✅ Plug-and-play via `BaseImbalanceHandler`

---

## ✅ Architecture Placement

```
ClassificationModelWrapper
↓
Pipeline:
Preprocessor
↓
SMOTEHandler ✅
↓
Model
```

---

## ✅ Key Features

| Feature               | Supported |
| --------------------- | --------- |
| Oversampling          | ✅        |
| Undersampling         | ❌        |
| Pipeline Integration  | ✅        |
| CV-safe execution     | ✅        |
| Before/After tracking | ✅        |
| Config-driven         | ✅        |

---

## ✅ Important Notes

✅ SMOTE applies ONLY during training

- ✅ Applied during pipeline.fit()
- ❌ NOT applied during predict()
- ✅ CV-safe when using imblearn.pipeline.Pipeline

## ✅ SMOTE performs ONLY oversampling

| Operation            | Supported |
| -------------------- | --------- |
| Oversample minority  | ✅        |
| Undersample majority | ❌        |
| Remove noise         | ❌        |

## ✅ When to Use SMOTE

| Scenario           | Recommendation    |
| ------------------ | ----------------- |
| Moderate imbalance | ✅SMOTE           |
| Severe imbalance   | ✅SMOTE           |
| Clean dataset      | ✅SMOTE           |
| Noisy dataset      | ⚠️Prefer SMOTEENN |

## ✅ Limitations

- ❌ Does not handle categorical features directly
- ❌ May introduce synthetic noise in overlapping classes
- ❌ Not suitable for multilabel datasets (without adaptation)

## ✅ Future Extensions

This handler is designed to support:

- ✅ ADASYN
- ✅ SMOTEENN
- ✅ SMOTETomek
- ✅ Custom imbalance strategies

## ✅ Summary

SMOTEHandler provides:

- ✅ Clean pipeline integration
- ✅ No data leakage
- ✅ Full audit visibility
- ✅ Config-driven execution
- ✅ Extensible design
