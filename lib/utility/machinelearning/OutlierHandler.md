# OutlierHandler Documentation

## Overview

`OutlierHandler` is a custom scikit-learn compatible transformer designed to detect and handle outliers in numerical data. It supports both IQR and Z-score methods and includes detailed logging for transparency.

---

## Features

- Supports two methods:
  - Interquartile Range (IQR)
  - Z-score
- Works seamlessly inside scikit-learn pipelines
- Does NOT remove rows (only modifies values)
- Tracks:
  - Outliers before handling
  - Outliers after handling
  - Configuration used

---

## Initialization

```python
OutlierHandler(method="iqr", factor=1.5, z_thresh=3)
```

### Parameters

- `method`: "iqr" or "zscore"
- `factor`: Multiplier for IQR (default = 1.5)
- `z_thresh`: Z-score threshold (default = 3)

---

## Methods

### 1. fit(X, y=None)

Learns outlier boundaries from the data.

#### IQR Method

- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1

Bounds:

- Lower = Q1 - factor × IQR
- Upper = Q3 + factor × IQR

Stores:

- Bounds per column
- Count of outliers before handling

#### Z-score Method

- Mean and standard deviation per column
- Computes z-scores

Stores:

- Mean and std per column
- Count of outliers where |z| > threshold

---

### 2. transform(X)

Applies outlier handling.

#### IQR Method

- Caps values below lower bound → lower
- Caps values above upper bound → upper

#### Z-score Method

- Replaces outliers with mean

---

## Logging (results attribute)

The class maintains a dictionary:

```python
self.results
```

### Structure:

```python
{
  "outliers_before": {
    "col1": 10,
    "col2": 5
  },
  "outliers_after": {
    "col1": {"before": 10, "after": 0},
    "col2": {"before": 5, "after": 0}
  },
  "config": {
    "method": "iqr",
    "factor": 1.5,
    "z_thresh": 3
  }
}
```

---

## Important Behavior

- ✅ Dataset shape remains unchanged
- ✅ Only numeric columns are processed
- ✅ Missing columns are safely ignored during transform
- ✅ Safe for training and prediction phases

---

## Usage Example

```python
outlier = OutlierHandler(method="iqr")
X_clean = outlier.fit_transform(X)

print(outlier.results)
```

---

## Pipeline Usage

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('outlier', OutlierHandler()),
    ('model', LinearRegression())
])
```

Access results inside pipeline:

```python
pipeline.named_steps['outlier'].results
```

---

## Limitations

- Does not remove rows
- Only handles numeric data
- Z-score may be affected by extreme values

---

## Best Practices

- Use IQR for skewed distributions
- Use Z-score for normal distributions
- Apply after missing value imputation
- Always validate results using logs

---

## Summary

`OutlierHandler` provides a flexible and pipeline-friendly way to manage outliers without breaking dataset integrity. It improves model robustness while keeping preprocessing transparent through detailed logs.
