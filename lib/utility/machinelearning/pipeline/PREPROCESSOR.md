
# Preprocessor – Detailed Documentation

## Overview

The `Preprocessor` class is responsible for building a reusable and consistent data preprocessing pipeline using scikit-learn components. It integrates numerical scaling, categorical encoding, and optional preprocessing steps such as imputation and outlier handling.

---

## Purpose

- Standardize preprocessing across all models
- Separate feature engineering from model logic
- Enable reusable pipelines
- Support plug-and-play preprocessing steps

---

## Architecture

```
Raw Data → (Imputer) → (OutlierHandler) → ColumnTransformer → Model
```

---

## Initialization

```python
Preprocessor(X, imputer=None, outlier_handler=None)
```

### Parameters

- **X**: Input feature DataFrame
- **imputer** *(optional)*: Custom imputation logic
- **outlier_handler** *(optional)*: Outlier processing logic

---

## Method: build()

```python
build() → Pipeline
```

Constructs a complete preprocessing pipeline.

---

## Step-by-Step Breakdown

### 1. Column Selection

```python
num_cols = X.select_dtypes(include=["int64", "float64"])
cat_cols = X.select_dtypes(include=["object", "category", "string"])
```

Separates:
- Numerical features
- Categorical features

---

### 2. Numeric Pipeline

```python
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])
```

Purpose:
- Normalize numeric features
- Improve model convergence

---

### 3. Categorical Pipeline

```python
categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

Purpose:
- Convert categorical variables into numerical form
- Handle unseen categories safely

---

### 4. Column Transformer

```python
ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])
```

Purpose:
- Apply transformations to correct column types
- Combine outputs into a unified feature matrix

---

### 5. Optional Steps

```python
steps = []

if imputer:
    steps.append(("imputer", imputer))

if outlier_handler:
    steps.append(("outlier", outlier_handler))
```

Provides flexibility for:
- Missing value handling
- Outlier correction

---

### 6. Final Pipeline

```python
steps.append(("column_transform", column_transform))
Pipeline(steps)
```

Final output is a complete preprocessing pipeline.

---

## Output

Returns:

```python
sklearn.pipeline.Pipeline
```

Ready to be integrated into model wrapper pipelines.

---

## Design Principles

- ✅ Modular design
- ✅ Reusability across models
- ✅ Separation of concerns
- ✅ Scikit-learn compatibility

---

## Integration

Used in:

- `ClassificationModelUtility`
- `LinearModelUtility`
- `ExperimentRunner`

---

## Benefits

- Consistent preprocessing
- Reduced code duplication
- Easy experimentation
- Plug-and-play pipeline structure

---

## Best Practices

- Always separate numeric and categorical logic
- Use `handle_unknown="ignore"` for production pipelines
- Keep preprocessing lightweight for efficiency
- Avoid data leakage (fit only on training data)

---

## Extensibility

You can extend this pipeline with:

- Feature selection
- Feature scaling alternatives (MinMaxScaler)
- Custom transformers
- Feature engineering steps

---

## Summary

The `Preprocessor` class is a foundational component of the ML pipeline that ensures consistent, reusable, and scalable data preparation across models.

