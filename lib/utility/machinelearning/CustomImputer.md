# CustomImputer

A flexible and reusable imputation transformer built using scikit-learn's BaseEstimator and TransformerMixin.

## 🚀 Features

- Supports numerical imputation (mean / median)
- Supports categorical imputation (mode)
- Group-based imputation support
- Automatic detection of numeric and categorical columns
- Built-in logging of before/after missing values
- Compatible with scikit-learn Pipelines

---

## 🧠 How It Works

### Step 1: Fit

- Detects numeric and categorical columns
- Computes global statistics (mean / median / mode)
- Logs missing values before imputation

### Step 2: Transform

- Applies group-based imputation (if groupby columns provided)
- Applies fallback global imputation
- Logs missing values after imputation

---

## 📦 Installation

This class requires:

```bash
pip install pandas scikit-learn
```

---

## 📥 Usage

### Import

```python
from your_file import CustomImputer
```

### Example

```python
imputer = CustomImputer(
    num_strategy="mean",
    cat_strategy="mode",
    groupby_cols=["Education", "Marital_Status"]
)

X_clean = imputer.fit_transform(X)
```

---

## ⚙️ Parameters

| Parameter    | Description                               |
| ------------ | ----------------------------------------- |
| num_strategy | "mean" or "median" for numeric columns    |
| cat_strategy | "mode" for categorical columns            |
| groupby_cols | List of columns for group-wise imputation |

---

## 📊 Logging Output

All logs are stored in:

```python
imputer.results
```

### Example:

```python
{
  "imputation_details_before": {...},
  "imputation_details_after": {...},
  "config": {...}
}
```

---

## ✅ Key Benefits

- Reduces bias with group-based imputation
- Ensures no missing values remain (fallback applied)
- Works seamlessly inside ML pipelines
- Improves model performance with better data quality

---

## 🔌 Pipeline Integration

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', CustomImputer()),
    ('model', LinearRegression())
])
```

---

## 📜 License

Free to use for learning and projects 🚀
