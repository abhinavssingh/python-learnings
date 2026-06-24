# ClassificationModelUtility

## Overview

ClassificationModelUtility is a central orchestration layer that manages end-to-end classification workflows in a unified, extensible, and production-ready manner. It orchestrates the complete lifecycle of machine learning experiments—from data preparation to model evaluation and result standardization—through a clean, modular architecture.
The utility coordinates:

- Data preparation and preprocessing
- Wrapper-driven model execution
- Configurable pipeline construction
- Imbalance handling (e.g., SMOTE)
- Ensemble modeling (parallel, sequential, stacking)
- Hyperparameter tuning
- Metrics, artifacts, and result standardization via ResultBuilder

---

## Architecture

### Summary

```
Data → Feature Selection
        ↓
Split (Pipeline Script)
        ↓
Utility (Orchestration Only)
        ↓
Preprocessor (Pipeline-Bound)
        ↓
Wrapper → Pipeline
        ↓
SMOTE / Imbalance
        ↓
Model / Ensemble
        ↓
Metrics + Artifacts
        ↓
ResultBuilder
        ↓
Model Save + Validation
        ↓
Results Store / Visualization
        ↓
Inference Pipeline (Decoupled)
```

### DATA PREPARATION FLOW

```mermaid
flowchart TD

    A[Input Dataset] --> B[Feature Selection / Cleaning]

    B --> C[Pipeline Script]

    C --> D[Train-Test Split]
    D --> E[X_train, X_test, y_train, y_test]

    E --> F[ClassificationModelUtility.prepare_data]

    F --> G["Label Encoding (Train Only)"]
    G --> H[Problem Type Detection]

    H --> I[Feature Validation]
    I --> J["Preprocessor Build (Pipeline Only)"]

    J --> K["Feature Pipeline (Imputer + Encoder + Scaling)"]

    K --> L[Reusable Preprocessor]
```

### MODEL EXECUTION FLOW (Utility Layer)

```mermaid
flowchart TD

    A[User Pipeline Script] --> B["Utility Layer (Classification / Regression / Unsupervised)"]

    B --> C[ModelRegistry]
    B --> D["Preprocessor (Pre-built)"]

    C --> E[Fetch Model Wrapper]
    E --> F[Deep Copy Wrapper]

    F --> G["Inject Imbalance Handler (SMOTE)"]

    G --> H[Wrapper.build_pipeline]

    H --> I[Pipeline Construction]

    I --> I1[Preprocessor]
    I1 --> I2["SMOTE / Resampling"]
    I2 --> I3[Model / Ensemble]

    I3 --> J["Pipeline.fit (Train Only)"]

    J --> K["Predict (X_test)"]
    K --> L["Predict Proba (Safe)"]

    L --> M[Metrics Evaluation]

    M --> N[Artifact Extraction]

    N --> O[ResultBuilder]

    O --> P[Results Store]

    P --> Q[Trained Models Registry]
```

### TRAINING & INFERENCE FLOW

```mermaid
flowchart TD

    A[Pipeline Script] --> B[Split Data]
    B --> C[Utility Initialization]

    C --> D["Prepare Data (No Split)"]

    D --> E[Get Wrapper from Registry]
    E --> F[Deep Copy Wrapper]

    F --> G{Multilabel?}
    G -->|Yes| H[Wrap with OneVsRestClassifier]
    G -->|No| I[Continue]

    H --> J["Inject SMOTE (if applicable)"]
    I --> J

    J --> K[Build Pipeline]

    K --> L["Pipeline = Preprocessor → SMOTE → Model"]

    L --> M["Train (fit)"]

    M --> N{Success?}
    N -->|Yes| O["Predict(X_test)"]
    N -->|No| P[Capture Error]

    O --> Q[Predict Proba]
    Q --> R[Metrics Evaluation]

    R --> S[ResultBuilder]

    S --> T["Save Model (pipeline.pkl + metadata)"]

    T --> U[Validate Inference]
```

### ARTIFACT EXTRACTION FLOW

```mermaid
flowchart TD

    A[Raw Metrics Output] --> B[Extract Artifacts]

    B --> C[ROC Curve]
    B --> D[PR Curve]
    B --> E[Classification Report]

    B --> F{Multilabel?}

    F -->|No| G[Confusion Matrix]
    F -->|Yes| H[Skip CM]

    B --> I{SMOTE Applied?}

    I -->|Yes| J[Extract Imbalance Summary]
    I -->|No| K[Skip]

    J --> J1[Before vs After Distribution]

    C --> L[Artifacts Dict]
    D --> L
    E --> L
    G --> L
    J1 --> L

    A --> M[Numeric Metrics]
```

---

## ✅ Key Responsibilities

- ✅ Data preparation, preprocessing, and splitting
- ✅ Wrapper-based model execution
- ✅ Ensemble modeling:
  - Parallel (Voting, Bagging)
  - Sequential (Boosting)
  - Stacking (Meta-learning)
- ✅ Config-driven imbalance handling (SMOTE, future strategies)
- ✅ Pipeline construction:
  `Preprocessor → SMOTE → Model / Ensemble`
- ✅ Metric computation via Metrics.classification
- ✅ Artifact extraction:
  - ROC Curve
  - PR Curve
  - Confusion Matrix
- ✅ Imbalance summary
- ✅ Hyperparameter tuning (ClassificationHyperparameterTuner)
  ✅ Result standardization via ResultBuilder ✅
  ✅ Model comparison, ranking, and analysis
  ✅ Failure-safe execution (error isolation per model)

---

## 🚀 Initialization

```python

cm = ClassificationModelUtility(
    df,
    target_col,
    imputer=imputer,
    outlier_handler=outlier,
    imbalance_config=imbalance_config
)

```

P**arameters**

- **df**: Input dataset
- **target_col**: Target column or list (multilabel supported)
- **imputer**: Optional CustomImputer
- **outlier_handler**: Optional OutlierHandler
- imbalance_config

```python

{
    "type": "smote",
    "params": {
        "k_neighbors": 5,
        "sampling_strategy": "auto",
        "random_state": 42
    }
}
```

---

## 📊 Data Preparation

```python
cm.prepare_data()
```

### Features

- Auto-detects problem type (binary, multiclass, multilabel)
- Encodes target labels (safe mapping stored for interpretation) ✅
- Applies preprocessing components:
- Imputer
  - OutlierHandler
  - Encoders / transformers (via Preprocessor)
- Builds reusable Preprocessor pipeline
- Ensures compatibility with imblearn Pipeline (flattened later) ✅

---

### Supported Problem Types

| Type       | Strategy         |
| ---------- | ---------------- |
| Binary     | Train/Test Split |
| Multiclass | Train/Test Split |
| Multilabel | Iterative Split  |

---

## ⚙️ Running Experiments

```python
cm.run_experiment("LogisticRegression")
```

### Execution Flow

1. Retrieve model wrapper from registry
2. Deep copy wrapper (avoid state leakage)
3. Wrap with OneVsRest (for multilabel)
4. Build pipeline (Preprocessor + Model)
5. Train pipeline (`fit`)
6. Predict labels
7. Predict probabilities (safe handling)
8. Compute metrics via Metrics.classification
9. Extract artifacts
10. Store structured results

---

## 🔁 Run All Models

```python
cm.run_all_models()
```

### Output

- Returns structured DataFrame
- Skips failed models safely

---

## 📦 Artifact Handling (Refactored)

Artifacts include:

- ROC Curve
- PR Curve
- Confusion Matrix (only for non-multilabel)
- Classification Report

### Separation Logic

```python
artifacts, metrics = _extract_artifacts(metrics)
```

✔ Keeps metrics clean
✔ Avoids UI pollution

---

## 🎯 Probability Handling

Multilabel probabilities are normalized:

```python
np.column_stack([p[:, 1] for p in raw_proba])
```

---

## 🔧 Hyperparameter Tuning (Updated)

```python
cm.tune_model("RandomForest", param_grid)
```

### Supports

- Grid Search ✅
- Random Search ✅

### Key Fix

✔ Strict classification scoring

```
scoring = "accuracy" or "f1_weighted"
```

---

## 📊 Results & Reporting

### Get Results DataFrame

```python
cm.get_results_df()
```

### Get Artifacts

```python
cm.get_artifacts_df()
```

---

## 🧠 Model Comparison

### Rank Models

```python
cm.rank_models(metric="f1")
```

### Best Model

```python
cm.get_best_model(metric="roc_auc")
```

---

## 📉 Confusion Matrix Handling

```python
cm.get_confusion_matrix_df(model_name)
```

✔ Uses ClassificationFormatter
✔ Avoids multilabel misuse

---

## ⚙️ Internal Helpers

### \_get_probabilities

Handles:

- Binary models
- Multiclass
- Multilabel

---

### extract_artifacts

Splits metrics vs artifacts

---

## ✅ Design Principles (Updated)

- ✅ Wrapper-based architecture
- ✅ Strict task separation (classification vs regression)
- ✅ Pipeline consistency
- ✅ Artifact-aware design
- ✅ Fault-tolerant execution
- ✅ AutoML-ready structure

---

## ✅ Best Practices

- Avoid confusion matrix for multilabel
- Always validate label distribution
- Use formatter layer for visualization
- Keep metrics computation pure
- Always use wrapper.evaluate()

---

## 🚀 Extensibility

Future Enhancements:

- AutoML orchestration layer
- Model explainability (SHAP)
- Threshold optimization
- Ensemble models (stacking/voting)

---

## ✅ Final Summary

`ClassificationModelUtility` now acts as:

🚀 A fully modular, wrapper-driven ML orchestration engine

Supporting:

- ✅ Multi-model execution
- ✅ Multiclass & multilabel
- ✅ Hyperparameter tuning
- ✅ Artifact-driven evaluation
- ✅ Production-grade ML pipelines
