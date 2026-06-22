
# DimensionalityPlots Documentation

## ✅ Overview

This module provides visualization utilities for **dimensionality reduction techniques**.

It supports:

- ✅ PCA (Principal Component Analysis)
- ✅ t-SNE (t-distributed Stochastic Neighbor Embedding)

These plots are primarily used for:

- Visualizing high-dimensional data
- Understanding clustering behavior
- Inspecting class separation
- Debugging unsupervised learning pipelines

---

## ✅ Architecture

```
Pipeline / Model Utility
        ↓
Preprocessing → Dimensionality Reduction
        ↓
DimensionalityPlots
        ↓
Plotly Visualizations
```

---

## ✅ Dependencies

```python
import pandas as pd
import plotly.express as px
```

---

## ✅ Plot Functions

---

### ✅ 1. PCA 2D Projection

```python
plot_pca_2d(X_reduced, labels=None)
```

#### Description
- Visualizes 2D PCA-transformed data
- Colors points based on labels (if provided)

#### Input
- `X_reduced`: 2D array (n_samples × 2)
- `labels`: Optional cluster/class labels

#### Output
- Scatter plot (PC1 vs PC2)

---

### ✅ 2. PCA 3D Projection

```python
plot_pca_3d(X_reduced, labels=None)
```

#### Description
- Visualizes 3D PCA-transformed data
- Useful for deeper structure inspection

#### Input
- `X_reduced`: 3D array (n_samples × 3)
- `labels`: Optional labels

#### Output
- 3D scatter plot (PC1, PC2, PC3)

---

### ✅ 3. t-SNE 2D Projection

```python
plot_tsne_2d(X_embedded, labels=None)
```

#### Description
- Non-linear dimensionality reduction visualization
- Captures local structure (clusters)

#### Input
- `X_embedded`: 2D t-SNE output
- `labels`: Optional cluster/class labels

#### Output
- Scatter plot (Dim1 vs Dim2)

---

## ✅ Design Highlights

- ✅ Supports both supervised and unsupervised use cases
- ✅ Handles optional labels gracefully
- ✅ Clean and minimal interface
- ✅ Fully compatible with Plotly for interactivity
- ✅ Ideal for integration into dashboards

---

## ✅ Best Practices

- Use PCA for:
  - fast visualization
  - variance explanation

- Use t-SNE for:
  - cluster visualization
  - non-linear patterns

- Always scale/normalize data before applying PCA/t-SNE

---

## ✅ Example Usage

```python
X_pca = pca.fit_transform(X)
fig = plot_pca_2d(X_pca, labels)
fig.show()

X_tsne = tsne.fit_transform(X)
fig = plot_tsne_2d(X_tsne, labels)
fig.show()
```

---

## ✅ Summary

The `DimensionalityPlots` module provides essential visualization tools for exploring high-dimensional datasets.

It enables:

- ✅ intuitive understanding of data structure
- ✅ visual cluster validation
- ✅ debugging and insight generation

and is a key component for **unsupervised and exploratory analysis workflows**.

