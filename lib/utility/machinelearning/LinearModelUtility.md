# LinearModelUtility Documentation

## Overview

`LinearModelUtility` is a reusable, beginner-friendly machine learning utility class designed to simplify the end-to-end workflow for regression tasks using scikit-learn.

## 🚀 Features

- ✅ Automatic data preprocessing (scaling + encoding)
- ✅ Train single, multiple, or all models
- ✅ Built-in evaluation (MSE, R²)
- ✅ K-Fold cross-validation support
- ✅ Hyperparameter tuning via GridSearchCV
- ✅ Flexible pipeline integration
- ✅ Centralized result tracking

##📦 Supported Models
The utility includes the following regression models:

- 📈 LinearRegression
- ⚙️ SGDRegressor
- 🔵 Ridge (L2 Regularization)
- 🔶 Lasso (L1 Regularization)
- ⚖️ ElasticNet (L1 + L2)

---

## 🧠 How It Works

The class automates the ML pipeline:

```
Raw Data
   ↓
Train-Test Split
   ↓
Preprocessing
   • Optional Imputer
   • Optional Outlier Handling
   • Scaling (StandardScaler)
   • Encoding (OneHotEncoder)
   ↓
Model Training
   ↓
Evaluation / Cross-Validation / Tuning
```

---

## 1. Initialization

```python
ml = LinearModelUtility(df, target_col)
```

- `df`: Input dataframe
- `target_col`: Target variable name

---

## 2. Data Splitting

```python
ml.split_data()
```

- Splits into train/test (80/20)
- Automatically detects:
  - Numerical columns
  - Categorical columns

Logs:

- Train/test shapes
- Column types

---

## 3. Preprocessing Pipeline

```python
ml.build_preprocessor(imputer=None, outlier_handler=None)
```

### Steps:

1. Optional Imputer
2. Optional Outlier Handler
3. ColumnTransformer:
   - Numeric → StandardScaler
   - Categorical → OneHotEncoder

---

## 4. Model Training

### A. Train Single Model

```python
ml.train_one("Ridge")
```

### B. Train Multiple Models

```python
ml.train_selected(["Ridge", "Lasso"])
```

### C. Train All Models

```python
ml.train_all()
```

---

## 5. Training Modes

### 1. Train-Test Mode

- Fits model on training data
- Evaluates on test data

Metrics:

- MSE (Mean Squared Error)
- R2 Score

---

### 2. K-Fold Cross Validation

```python
ml.train_selected(["Ridge"], k_fold=5)
```

- Splits training data into K folds
- Returns:
  - Fold scores
  - Mean score
  - Standard deviation

---

### 3. Grid Search (Hyperparameter Tuning)

```python
ml.train_model("Ridge", use_grid=True, param_grid={"model__alpha": [0.1, 1, 10]})
```

- Uses GridSearchCV
- Returns best parameters and best estimator

---

## 6. Generic Tuning Method

```python
ml.tune("Ridge", {"model__alpha": [0.1, 1, 10]})
```

Outputs:

- Best parameters
- Best cross-validation score
- Best pipeline model

---

## 7. Results Storage

All outputs are stored in:

```python
ml.results
```

Structure includes:

- Data split details
- Preprocessing configuration
- Model metrics
- Cross-validation scores
- Best models

---

## 8. Model Descriptions

Each model includes an explanation:

- LinearRegression → baseline model
- SGDRegressor → large-scale optimized
- Ridge → L2 regularization
- Lasso → feature selection (L1)
- ElasticNet → hybrid regularization

---

## 9. Key Features

- Modular design
- Pipeline-based (avoids data leakage)
- Automatic preprocessing
- Flexible training modes
- Works with custom transformers (imputer, outlier handler)

---

## 10. Best Practices

- Always scale numeric features
- Use cross-validation for reliable evaluation
- Prefer tuning over default parameters
- Avoid dropping rows inside pipelines
- Access trained pipeline via results dictionary

---

## 11. Example Usage

```python
ml = LinearModelUtility(df, "target")

ml.train_all(imputer=my_imputer, outlier_handler=my_outlier)

print(ml.results)
```

---

## Summary

`LinearModelUtility` provides a clean, extensible framework for regression modeling. It abstracts repetitive ML tasks and ensures consistent preprocessing, training, and evaluation across models.
