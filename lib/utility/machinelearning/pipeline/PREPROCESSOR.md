# Preprocessor (Updated Documentation)

## ✅ Overview

Preprocessor is a central pipeline builder responsible for constructing a consistent, reusable, and configurable feature transformation pipeline across all machine learning workflows.
It supports both:

- ✅ Supervised learning
- ✅ Unsupervised learning

using a mode-driven design.

---

## 🧱 Updated Architecture Flow

```mermaid
flowchart TD
    A[Raw Data] --> B[CustomImputer]
    B --> C[OutlierHandler]
    C --> D[ColumnTransformer]

    D --> E[Numeric Pipeline]
    D --> F[Categorical Pipeline]

    E --> E1[StandardScaler]

    F --> F1[Encoder Selection]
    F1 --> F2["OneHotEncoder (Supervised)"]
    F1 --> F3["OrdinalEncoder (Unsupervised)"]

    E1 --> G[Combined Features]
    F2 --> G
    F3 --> G

    G --> H[Processed Data]
```

---

## ✅ Key Responsibilities

- ✅ Build preprocessing pipeline (NOT execute transformations directly)
- ✅ Integrate imputation and outlier handling
- ✅ Separate numeric and categorical transformations
- ✅ Dynamically adjust encoding strategy based on mode
- ✅ Ensure compatibility with Wrapper + Utility layers

---

## ⚙️ Constructor

```python
Preprocessor(X, imputer=None, outlier_handler=None,mode="supervised" )
```

---

## 🔧 Build Method

```python
pipeline = preprocessor.build()
```

---

## 🔍 Pipeline Steps (Updated)

### 1. Optional Steps

```
CustomImputer → OutlierHandler
```

### 2. ColumnTransformer

- Numeric → StandardScaler
- Categorical → OneHotEncoder

---

## Why Mode Matters

### Supervised Mode

OneHotEncoder

    Preserves categorical independence
    Ideal for tree + linear models

### Unsupervised Mode

OrdinalEncoder

    Prevents high dimensionality explosion
    Maintains distance consistency
    Critical for clustering algorithms

---

## ✅ Column Detection (Improved)

```
include="number"
include=["object", "category", "string"]
```

---

## ✅ Output

Returns:

```
sklearn.pipeline.Pipeline
```

Used inside Wrapper:

```
wrapper.build_pipeline(preprocessor)
```

---

## ✅ Framework Integration

Used in:

- ClassificationModelUtility
- LinearModelUtility
- ExperimentRunner

---

## ✅ Design Principles (Updated)

- ✅ Wrapper-driven design
- ✅ Separation from model logic
- ✅ Reusable pipeline
- ✅ No data leakage
- ✅ Plug-and-play components

---

## ⚠️ Best Practices

- Fit only on training data
- Always pass same pipeline to wrapper
- Keep steps modular
- Avoid heavy transformations

---

## ✅ Final Summary

`Preprocessor` is the **foundation of your ML pipeline**, ensuring consistent, reusable, and production-ready feature processing.
