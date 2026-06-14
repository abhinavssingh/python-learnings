# OutlierHandler (Updated Documentation)

## ✅ Overview

`OutlierHandler` is a **pipeline-compatible transformer** for detecting and handling outliers in numerical data.

---

## 🚀 Key Features (Updated)

- Supports IQR & Z-score methods
- Uses clipping (IQR) for stability
- No row removal (safe transformations)
- Numeric-only processing
- Logging before & after handling
- Fully pipeline compatible

---

## 🧱 Architecture Role

```
CustomImputer → OutlierHandler → Feature Encoding → Model
```

---

## ⚙️ Parameters

| Parameter | Description       |
| --------- | ----------------- |
| method    | "iqr" or "zscore" |
| factor    | IQR multiplier    |
| z_thresh  | Z-score threshold |

---

## 🔍 Internal Logic

### ✅ IQR Method

- Lower = Q1 - factor \* IQR
- Upper = Q3 + factor \* IQR
- Uses `.clip()` to cap values

### ✅ Z-score Method

- Replace outliers with mean
- Skips std=0 cases

---

## 📊 Logging

```
self.results_
```

```
{
  "outliers_before": {...},
  "outliers_after": {...},
  "config": {...}
}
```

---

## ✅ Framework Alignment

- ✅ Works inside Preprocessor
- ✅ Safe for pipelines
- ✅ No shape change
- ✅ Works across train/test

---

## ⚠️ Best Practices

- Apply after imputation
- Use IQR for skewed data
- Use Z-score for normal distributions
- Always check logs

---

## ✅ Final Summary

`OutlierHandler` ensures **robust outlier handling without breaking dataset integrity**.
