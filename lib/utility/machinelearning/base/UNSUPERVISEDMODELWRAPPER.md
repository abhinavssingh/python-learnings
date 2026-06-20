# UnsupervisedModelWrapper Documentation

## ✅ Overview

`UnsupervisedModelWrapper` is a base wrapper class for all unsupervised learning models. It provides a unified interface for:

- Pipeline construction
- Model execution (fit + predict)
- Evaluation using standardized metrics
- Integration with the broader ML framework

---

## ✅ Class Definition

```python
class UnsupervisedModelWrapper(BaseModelWrapper):
```

---

## ✅ Key Responsibilities

- Build preprocessing + model pipeline
- Execute clustering or dimensionality models
- Standardize evaluation using `Metrics.unsupervised`
- Ensure compatibility with ResultBuilder and Utility layer

---

## ✅ Attributes

| Attribute | Description                                |
| --------- | ------------------------------------------ |
| task      | "unsupervised"                             |
| family    | "unsupervised" (extended in child classes) |

---

## ✅ Pipeline Construction

```python
def build_pipeline(self, preprocessor):
```

### ✅ Flow

```
Preprocessor → Model
```

### ✅ Logic

- Flattens preprocessor if it already contains steps
- Appends the model as final pipeline step
- Uses `imblearn.pipeline.Pipeline`

---

## ✅ Model Execution

```python
def predict(self, X):
```

### ✅ Behavior

| Case                 | Action              |
| -------------------- | ------------------- |
| `fit_predict` exists | Uses it directly ✅ |
| `predict` exists     | Calls after fit     |
| `transform` exists   | Used for PCA/t-SNE  |

### ✅ Output

- Clustering → labels
- Dimensionality → embeddings

---

## ✅ Evaluation

```python
def evaluate(self, X, labels):
```

### ✅ Key Feature

- Extracts processed data from pipeline without refitting
- Ensures correct metric computation on transformed data

### ✅ Metrics Used

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

---

## ✅ Pipeline Flow

```
Raw Data
   ↓
Preprocessor
   ↓
Pipeline.fit_predict()
   ↓
Output (labels / embeddings)
   ↓
Evaluation (Metrics.unsupervised)
```

---

## ✅ Design Highlights

- ✅ Pipeline-first architecture
- ✅ Reusable across all unsupervised models
- ✅ Supports clustering and dimensionality reduction
- ✅ No data leakage (transform without refit)
- ✅ Fully compatible with ResultBuilder and Utility layers

---

## ✅ Example Usage

```python
wrapper = KMeansWrapper(n_clusters=3)
wrapper.build_pipeline(preprocessor)
labels = wrapper.predict(X)
metrics = wrapper.evaluate(X, labels)
```

---

## ✅ Final Takeaway

> `UnsupervisedModelWrapper` provides a unified, extensible, and production-ready interface for executing unsupervised machine learning workflows.
