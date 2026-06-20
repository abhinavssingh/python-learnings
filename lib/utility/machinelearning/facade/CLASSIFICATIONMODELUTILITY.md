# ClassificationModelUtility

## Overview

The ClassificationModelUtility is a high-level orchestration layer that manages end-to-end classification workflows in a unified and extensible manner. It coordinates data preparation, wrapper-based model execution, pipeline construction, imbalance handling (e.g., SMOTE), evaluation, hyperparameter tuning, and result management.
The utility leverages a modular architecture built on top of:

- Wrapper-based model abstraction (BaseModelWrapper, ClassificationModelWrapper)
- Config-driven pipeline composition (Preprocessor → Imbalance Handler → Model)
- Centralized model discovery via ModelRegistry

It ensures:

- ✅ Clean separation of concerns
- ✅ CV-safe and pipeline-driven execution
- ✅ Plug-and-play extensibility for new models and strategies
- ✅ Built-in observability (metrics, artifacts, imbalance tracking)
  By combining these capabilities, ClassificationModelUtility acts as the core orchestration engine, enabling scalable experimentation, consistent evaluation, and production-grade classification pipelines.rk.

---

## Architecture

### DATA PREPARATION FLOW

```mermaid
flowchart TD
    A[Input Dataset] --> B[DataLoader]
    B --> C[DataFrameHelper]

    C --> D[ClassificationModelUtility.prepare_data]

    D --> E[Imputer]
    E --> F[OutlierHandler]

    F --> G[Target Extraction]

    G --> H[Label Encoding]
    H --> I[Problem Type Detection]

    I --> J{Problem Type}

    J -->|Binary / Multiclass| K[Train-Test Split]
    J -->|Multilabel| L[Iterative Train-Test Split]

    K --> M[X_train, X_test, y_train, y_test]
    L --> M

    M --> N[Preprocessor Build]

    N --> O["Feature Transformers Pipeline (Imputer + Encoder + Scaling)"]

    O --> P[Ready for Model Pipeline ✅]
```

### MODEL EXECUTION FLOW (Utility Layer)

```mermaid
flowchart TD
    A[User API / Script] --> B[ClassificationModelUtility]

    B --> C[ModelRegistry]
    B --> D[Preprocessor]

    C --> E[Fetch Model Wrapper]
    E --> F[Deep Copy Wrapper ✅]

    F --> G["Inject Imbalance Handler (SMOTE) ✅"]

    G --> H[ClassificationModelWrapper.build_pipeline]

    H --> I[Flatten Preprocessor Steps ✅]

    I --> J[Pipeline Construction]

    J --> J1[Preprocessor Steps]
    J1 --> J2["SMOTE (fit_resample) ✅"]
    J2 --> J3[Model]

    J3 --> K[Train Pipeline]
    K --> L[Predict]

    L --> M[Predict Proba]

    M --> N[Metrics.classification]

    N --> O[Artifacts Extraction]

    O --> P[Results Store ✅]
```

### TRAINING & INFERENCE FLOW

```mermaid
flowchart TD

    A[run_experiment] --> B[Get Wrapper from Registry]

    B --> C[Deep Copy Wrapper]

    C --> D{Multilabel?}
    D -->|Yes| E[Wrap with OneVsRestClassifier]
    D -->|No| F[Continue]

    E --> G["Inject SMOTE (Skip if multilabel ✅)"]
    F --> G

    G --> H[Build Pipeline]

    H --> I[Flatten Preprocessor Steps ✅]

    I --> J[Pipeline = Preprocessor → SMOTE → Model ✅]

    J --> K[Train Wrapper]
    K --> L["Pipeline.fit (Train Only)"]

    L --> M{Training Success?}

    M -->|Yes| N["Predict(X_test)"]
    M -->|No| O[Capture Error]

    N --> P[Predict Proba]

    P --> Q[Evaluate Metrics]
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

    I -->|Yes| J[Extract Imbalance Summary ✅]
    I -->|No| K[Skip Imbalance Info]

    J --> J1[Before vs After Distribution]

    C --> L[Artifacts Dict]
    D --> L
    E --> L
    G --> L
    J1 --> L

    A --> M[Numeric Metrics]
```

---

## ✅ Key Responsibilities (Refactored)

- Data preparation, preprocessing, and splitting
- Dynamic model execution via Wrapper-based architecture
- Config-driven imbalance handling (SMOTE / future strategies)
- Pipeline construction with flattened preprocessing + SMOTE + model
- Metric computation using Metrics.classification (classification-specific)
- Artifact extraction:
  - ROC Curve
  - PR Curve
  - Confusion Matrix (conditional)
  - Imbalance summary (before vs after SMOTE) ✅

- Hyperparameter tuning via ClassificationHyperparameterTuner
- Result aggregation, normalization, ranking, and comparison
- Failure-safe execution (graceful error capture per model)

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
