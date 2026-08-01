# Preprocessing

Framework-independent preprocessing utilities for ML and deep learning projects.

## File

- data_preprocessor.py

## Core Capability

DataPreprocessor.prepare_classification_data provides a reusable workflow for:

- Optional column dropping
- Binary label encoding
- One-hot encoding
- Numeric imputation and scaling
- Stratified train/test split

## Return Object

PreprocessingResult includes:

- X_train, X_test
- y_train, y_test
- fitted preprocessor
- label_encoders

## Typical Usage

```python
result = DataPreprocessor.prepare_classification_data(
    df=df,
    target_column="target",
    drop_columns=["id"],
    label_encode_columns=["binary_col"],
    one_hot_columns=["multi_class_col"],
    scale_numeric=True,
    test_size=0.2,
    random_state=42,
)
```

Use transform_new_data to preprocess inference records with the fitted encoders and transformer.
