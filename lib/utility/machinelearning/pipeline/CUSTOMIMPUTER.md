# CustomImputer (Updated Documentation)

## ✅ Overview

`CustomImputer` is a **wrapper-aligned preprocessing component** designed for robust missing value handling in ML pipelines.

It is fully compatible with:

- ✅ sklearn pipelines
- ✅ Preprocessor layer
- ✅ Classification & Regression utilities

---

## 🚀 Key Features (Updated)

- Supports numerical strategies: mean / median
- Supports categorical strategies: mode / constant
- Group-based imputation (advanced)
- Automatic dtype detection
- Safe fallback (no nulls remain)
- Handles non-DataFrame input
- Pipeline-safe transformation
- Built-in logging (`results_`)

---

## 🧱 Architecture Role

```
Raw Data → CustomImputer → OutlierHandler → ColumnTransformer → Model
```

---

## ⚙️ Parameters

| Parameter    | Description                            |
| ------------ | -------------------------------------- |
| num_strategy | "mean" or "median"                     |
| cat_strategy | "mode" or "constant"                   |
| groupby_cols | list of columns for grouped imputation |

---

## 🔍 Internal Logic

### ✅ Fit Phase

- Detect numeric & categorical columns
- Compute global fallback values
- Validate groupby columns
- Log missing values before processing

### ✅ Transform Phase

1. Apply group-based imputation (if provided)
2. Apply global fallback
3. Log missing values after imputation

---

## 📊 Logging Structure

```
imputer.results_
```

```
{
  "imputation_details_before": {...},
  "imputation_details_after": {...},
  "config": {...}
}
```

---

## ✅ Framework Alignment

- ✅ Works inside Preprocessor
- ✅ Stateless transformation per fit
- ✅ Compatible with wrapper pipelines
- ✅ Safe for train/test split

---

## ⚠️ Best Practices

- Always run before outlier handling
- Use groupby_cols for domain-aware imputation
- Avoid constant strategy unless necessary
- Validate results using logs

---

## ✅ Final Summary

`CustomImputer` is a **production-grade imputation layer** that improves data quality while maintaining pipeline consistency.
