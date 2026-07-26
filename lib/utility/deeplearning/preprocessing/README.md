# Data Preprocessor

A reusable, framework-independent preprocessing utility designed for Machine Learning and Deep Learning projects.

The DataPreprocessor provides a standardized approach for:

- Data Cleaning
- Feature Encoding
- Missing Value Handling
- Feature Scaling
- Train/Test Splitting
- Transformation of New Data
- Reusable Inference Pipelines

The goal is to eliminate repetitive preprocessing code and create a single preprocessing engine that can be used across TensorFlow, PyTorch, Scikit-Learn, XGBoost, and future projects.

---

# Folder Structure

```text
deeplearning/
│
├── preprocessing
│   │
│   ├── data_preprocessor.py
│   └── README.md
```

---

# Overview

Most machine learning projects require similar steps:

```text
Raw Dataset
      │
      ▼
Column Selection
      │
      ▼
Missing Value Handling
      │
      ▼
Categorical Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
```

Instead of implementing those steps separately in each project, `DataPreprocessor` centralizes them into a reusable component.

---

# Key Components

## DataPreprocessor

Main preprocessing engine.

Responsibilities:

- Dataset preparation
- Feature engineering
- Encoding categorical variables
- Scaling numerical variables
- Splitting data
- Preparing new records for inference

---

## PreprocessingResult

Container object returned after preprocessing.

Stores:

```python
X_train
X_test

y_train
y_test

preprocessor

label_encoders
```

Example:

```python
result = DataPreprocessor.prepare_classification_data(...)
```

Access:

```python
result.X_train
result.X_test

result.y_train
result.y_test

result.preprocessor
result.label_encoders
```

---

# Supported Features

## Column Removal

Remove irrelevant columns.

Common examples:

```text
CustomerId
EmployeeId
Surname
RowNumber
Timestamp
```

Example:

```python
drop_columns=[
    "CustomerId",
    "Surname",
]
```

---

## Label Encoding

Encodes binary categorical variables.

Example:

```text
Male
Female
```

becomes

```text
0
1
```

Configuration:

```python
label_encode_columns=[
    "Gender"
]
```

---

## One-Hot Encoding

Encodes multi-category variables.

Example:

```text
France
Germany
Spain
```

becomes

```text
France Germany
1      0
0      1
0      0
```

Configuration:

```python
one_hot_columns=[
    "Geography"
]
```

---

## Missing Value Handling

Uses:

```python
SimpleImputer
```

Current Strategy:

```python
median
```

Example:

```python
SimpleImputer(
    strategy="median"
)
```

Benefits:

- Handles missing numeric values
- Maintains dataset consistency
- Avoids training failures

---

## Feature Scaling

Uses:

```python
StandardScaler
```

Formula:

```text
(X - Mean) / Standard Deviation
```

Benefits:

- Faster convergence
- Better neural network performance
- More stable optimization
- Improved gradient descent behavior

---

## Train/Test Split

Automatically separates training and testing datasets.

Example:

```python
test_size=0.2
random_state=42
```

Output:

```text
80% Training
20% Testing
```

---

## Stratified Sampling

Classification datasets automatically preserve class distribution.

Example:

```python
stratify=y
```

Benefits:

- Balanced train set
- Balanced test set
- Reliable evaluation metrics

---

# Classification Example

## Bank Churn Prediction

```python
result = DataPreprocessor.prepare_classification_data(
    df=df,
    target_column="Exited",
    drop_columns=[
        "RowNumber",
        "CustomerId",
        "Surname",
    ],
    label_encode_columns=[
        "Gender",
    ],
    one_hot_columns=[
        "Geography",
    ],
    scale_numeric=True,
    test_size=0.2,
    random_state=0,
)
```

---

# Access Processed Data

```python
X_train = result.X_train
X_test = result.X_test

y_train = result.y_train
y_test = result.y_test
```

---

# TensorFlow Framework Integration

```python
model = MLPWrapper(
    input_dim=X_train.shape[1],
    output_dim=1,
    hidden_layers=[64, 32],
    output_activation="sigmoid",
)

utility = TensorFlowModelUtility(
    model_wrapper=model,
    config=config,
)

utility.compile()

utility.train(
    X_train,
    y_train,
)
```

---

# PyTorch Integration

Future implementation:

```python
model = TorchMLPWrapper(...)

trainer.train(
    X_train,
    y_train,
)
```

No preprocessing changes required.

---

# Future Data Prediction

Transform unseen customer records using the same preprocessing object used during training.

Example:

```python
new_customer = pd.DataFrame(...)
```

Transform:

```python
processed_customer = (
    DataPreprocessor.transform_new_data(
        new_customer,
        result.preprocessor,
        result.label_encoders,
    )
)
```

Predict:

```python
prediction = model.predict(
    processed_customer
)
```

This guarantees that training and inference use identical preprocessing logic.

---

# Workflow Example

```text
Raw Data
     │
     ▼
DataPreprocessor
     │
     ▼
PreprocessingResult
     │
     ├── X_train
     ├── X_test
     ├── y_train
     ├── y_test
     ├── Encoders
     └── Preprocessor
              │
              ▼
Model Wrapper
              │
              ▼
Trainer
              │
              ▼
Evaluator
```

---

# Supported Project Types

## Classification

Examples:

- Bank Churn Prediction
- Customer Segmentation
- Sentiment Analysis
- Fraud Detection
- Medical Diagnosis

---

## Regression

Examples:

- House Price Prediction
- Revenue Forecasting
- Sales Forecasting
- Demand Prediction

---

## Deep Learning

Examples:

- TensorFlow
- PyTorch

---

## Traditional Machine Learning

Examples:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

---

# Current Capabilities

Supported:

✅ Drop Columns

✅ Label Encoding

✅ One-Hot Encoding

✅ Missing Value Handling

✅ Feature Scaling

✅ Train/Test Split

✅ Stratified Sampling

✅ Future Data Transformation

✅ Classification Workflows

---

# Future Enhancements

Planned:

- Feature Selection
- PCA
- SMOTE
- Outlier Detection
- Target Encoding
- K-Fold Cross Validation
- Time Series Split
- Image Preprocessing
- NLP Preprocessing
- Auto Feature Engineering
- Feature Store Integration

---

# Design Principles

The DataPreprocessor follows the framework design goals:

- Reusable
- Framework Independent
- Configurable
- Production Ready
- Training/Inference Consistent
- Easy Integration with TensorFlow and PyTorch

By centralizing preprocessing logic, projects become cleaner, easier to maintain, and significantly more reusable.
