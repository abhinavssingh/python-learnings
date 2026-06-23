# Inference Layer - Machine Learning Framework

## Overview

The Inference Layer is responsible for loading trained pipelines and executing predictions consistently across tasks (classification, regression, unsupervised).

---

## Key Principles

- Training and inference are strictly separated
- Only serialized pipeline is used at runtime
- Wrapper classes are NOT used during inference
- Metadata drives inference behavior

---

## Folder Structure

```
inference/
│
├── BaseInferencePipeline.py
├── ClassificationInference.py
├── RegressionInference.py
├── UnsupervisedInference.py
├── InferenceFactory.py
└── README.md
```

---

## Components

### 1. BaseInferencePipeline

- Handles common logic
- Input validation
- Feature alignment

### 2. ClassificationInference

- predict()
- predict_proba()
- predict_with_threshold()

### 3. RegressionInference

- predict()

### 4. UnsupervisedInference

- predict()
- fit_predict() fallback

### 5. InferenceFactory

- Loads pipeline + metadata
- Routes to correct inference class

---

## Saved Model Structure

```
saved_models/
├── model_X/
│   ├── pipeline.pkl
│   ├── metadata.json
│   ├── labels.npy (optional for clustering)
```

---

## Metadata Example

```json
{
  "model": "RandomForestClassifier",
  "task": "classification",
  "feature_names": ["age", "salary"],
  "threshold": 0.52,
  "experiment": "RandomForest | tuned"
}
```

---

## Usage

### Load Model

```python
from inference.InferenceFactory import InferenceFactory

model = InferenceFactory.load("saved_models/best_model")
```

### Predict

```python
preds = model.predict(X)
```

### Classification Only

```python
proba = model.predict_proba(X)
preds = model.predict_with_threshold(X)
```

---

## Design Decisions

| Decision                  | Reason                           |
| ------------------------- | -------------------------------- |
| Use joblib                | efficient pipeline serialization |
| Store metadata separately | flexible inference control       |
| Factory pattern           | extensible architecture          |

---

## Future Enhancements

- Model versioning
- Schema validation
- Batch inference (Parquet)
- FastAPI deployment

---

## Summary

The inference layer ensures consistent, scalable, and production-ready predictions using trained pipelines without dependency on training wrappers.
