# 🚀 Machine Learning Framework (Modular & Extensible)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Architecture](https://img.shields.io/badge/Design-Loose%20Coupling-green)
![Status](https://img.shields.io/badge/Production-Ready-success)

---

## 📌 Overview

This repository provides a **modular, scalable, and loosely coupled machine learning framework** designed for:

- Experiment-driven ML workflows
- Rapid prototyping
- Production-grade extensibility

---

## 🧭 Architecture Overview

### ML Framework Layered Architecture

```mermaid
flowchart TD

    A["MLModelBase<br/>Core abstraction layer"] --> B["BaseModelWrapper<br/>Generic wrapper contract"]

    B --> C["ClassificationModelWrapper<br/>Task-specific implementation"]

    C --> D["Concrete Wrappers<br/>Logistic, RF, XGB, SVC, etc."]

    D --> E["ModelRegistry<br/>Auto-discovery & management"]

    E --> F["Utility Layer<br/>ClassificationModelUtility"]

    F --> G["Visualizer<br/>Plots, ROC, Metrics Charts"]
```

### 🔷 High-Level Flow

```mermaid
flowchart TD

    A[User Script / API] --> B[ClassificationModelUtility]

    B --> C[ModelRegistry]
    B --> D[Preprocessor]

    C --> E[Fetch Wrapper]
    E --> F[Deep Copy Wrapper]

    D --> G[Preprocessing Pipeline]

    F --> H["Build Pipeline<br/>(Preprocessor + Model)"]

    H --> I["Train<br/>pipeline.fit"]

    I --> J{Training Success?}

    J -->|Yes| K[Predict]
    J -->|No| Z[Capture Error]

    K --> L[Predict Proba]

    L --> M[Metrics Evaluation]

    M --> N[Extract Artifacts]

    N --> O[Store Results]

    Z --> O
```

---

### 🔁 Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant CMU as ClassificationModelUtility
    participant MR as ModelRegistry
    participant W as Wrapper
    participant P as Pipeline
    participant MET as Metrics

    U->>CMU: run_all_models()

    loop For each model
        CMU->>MR: get_model(name)
        MR-->>CMU: Wrapper

        CMU->>W: clone (deepcopy)
        CMU->>W: build_pipeline()

        CMU->>W: train()
        W->>P: fit()

        alt Success
            CMU->>W: predict()
            CMU->>W: predict_proba()

            CMU->>W: evaluate()
            W->>MET: compute metrics

            CMU->>CMU: extract artifacts
            CMU->>CMU: append results

        else Failure
            CMU->>CMU: log error
            CMU->>CMU: append failed result
        end
    end

    CMU-->>U: Results DataFrame
```

### RESULTS & COMPARISON FLOW

```mermaid
flowchart TD

    A[Experiment Results List] --> B[Results DataFrame]
    A --> C[Artifacts DataFrame]

    B --> D[Model Comparator]

    D --> E[Rank Models]
    D --> F[Best Model]
    D --> G[Compare Models]

    E --> H[Sorted Results]
    F --> I[Top Model]
```

### VISUALIZATION FLOW

```mermaid
flowchart TD

    A[Results DataFrame] --> B[ClassificationPlots]

    B --> C[Bar Charts]
    B --> D[Multi-Metric Charts]
    B --> E[Best Model Visualization]

    A --> F[Artifacts]

    F --> G[ROC Curves]
    F --> H[Confusion Matrix Plots]

    C --> I[PlotRenderer]
    G --> I
```

### REPORT GENERATION FLOW

```mermaid
flowchart TD

    A[Plots + Data] --> B[HtmlBuilder]

    B --> C[Build Cards]
    B --> D[Build Grids]
    B --> E[Embed Charts]

    C --> F[HTML Document]

    F --> G[ReportUtils.save_html]

    G --> H[Saved HTML File]
    H --> I[Auto Open in Browser]
```

---

## 📂 Folder Structure

```
machinelearning/
│
├── README.md
│   # ✅ Updated: Wrapper-driven architecture + end-to-end flow + AutoML-ready
│
├── base/
│   ├── MLModelBase.py
│   │   # ✅ Core abstraction (train / predict / evaluate contract)
│   │
│   ├── BaseModelWrapper.py
│   │   # ✅ Central wrapper (pipeline + lifecycle execution)
│   │
│   ├── ClassificationModelWrapper.py
│   │   # ✅ Classification wrapper (predict_proba + classification metrics)
│   │
│   ├── LinearRegressionModelWrapper.py
│       # ✅ Regression wrapper (regression-only evaluation)
│
├── evaluation/
│   ├── Metrics.py
│   │   # ✅ PURE metrics layer (classification vs regression separated)
│   │
│   ├── METRICS.md
│   │   # ✅ Updated (artifact separation + task-aware metrics)
│   │
│   ├── ModelComparator.py
│   │   # ✅ Generic comparison logic (experiment-level comparison)
│   │
│   ├── ClassificationModelComparator.py
│       # ✅ Classification ranking + best model selection
│
├── facade/
│   ├── ClassificationModelUtility.py
│   │   # ✅ Classification orchestration (wrapper-driven + artifact-aware)
│   │
│   ├── CLASSIFICATIONMODELUTILITY.md
│   │   # ✅ Updated (pipeline + metrics + artifacts + tuning flow)
│   │
│   ├── LinearModelUtility.py
│   │   # ✅ Regression orchestration (correct scoring + wrapper-based)
│   │
│   ├── LinearModelUtility.md
│       # ✅ Updated (regression separation + tuning fix)
│
├── models/
│   ├── __init__.py
│   │   # ✅ Model discovery entry (auto-registration)
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   │   # ✅ Registers classification models
│   │   │
│   │   ├── LogisticRegressionWrapper.py
│   │   ├── DecisionTreeClassifierWrapper.py
│   │   ├── RandomForestWrapper.py
│   │   ├── KNNClassifierWrapper.py
│   │   ├── SVCWrapper.py
│   │       # ✅ All implement: task="classification", family="*"
│   │
│   ├── linear/
│       ├── __init__.py
│       │   # ✅ Registers regression models
│       │
│       ├── LinearRegressionWrapper.py
│       ├── RidgeWrapper.py
│       ├── LassoWrapper.py
│       ├── ElasticNetWrapper.py
│           # ✅ All implement: task="regression", family="linear"
│
├── pipeline/
│   ├── Preprocessor.py
│   │   # ✅ PIPELINE BUILDER (not executor)
│   │   # ✅ Combines: Imputer → Outlier → Transformer
│   │
│   ├── CustomImputer.py
│   │   # ✅ Missing value handler (group-aware + results_ logging)
│   │
│   ├── CustomImputer.md
│   │   # ✅ Updated (framework integration + pipeline safety)
│   │
│   ├── OutlierHandler.py
│   │   # ✅ Outlier handler (IQR/Z-score, no row deletion)
│   │
│   ├── OutlierHandler.md
│       # ✅ Updated (pipeline compatibility + logging)
│
├── registry/
│   ├── ModelRegistry.py
│       # ✅ Wrapper discovery engine
│       # ✅ Task-aware (classification / regression)
│       # ✅ Family-aware (linear / tree / boosting)
│
├── shared/
│   ├── DataCleaner.py
│   │   # ✅ CV results flattening + cleanup
│   │
│   ├── Formatter.py
│   │   # ✅ Experiment naming (standardized format)
│   │
│   ├── ClassificationFormatter.py
│       # ✅ Artifact formatting (ROC / PR / CM → DataFrame/UI)
│
├── tuning/
│   ├── HyperparameterTuner.py
│   │   # ✅ REGRESSION tuner
│   │   # ✅ FIXED: scoring = neg_mean_squared_error
│   │
│   ├── HyperparameterTuner.md
│   │   # ✅ Updated (regression-only tuning flow)
│   │
│   ├── ClassificationHyperparameterTuner.py
│       # ✅ Classification tuner
│       # ✅ Wrapper-based execution + artifact-aware output
│
├── visualization/
│   ├── README.md
│   │   # ✅ Visualization architecture overview
│   │
│   ├── core/
│   │   ├── MetricResolver.py
│   │       # ✅ Dynamically selects best metrics for plots
│   │
│   ├── generic/
│   │   ├── ClassificationPlots.py
│   │   │   # ✅ Core plots (ROC, PR, multi-metric, comparisons)
│   │   │
│   │   ├── CLASSIFICATIONPLOTS.md
│   │   │   # ✅ Visualization documentation
│   │   │
│   │   ├── ModelPerformanceVisualizer.py
│   │   │   # ✅ Visual orchestration layer (results + artifacts)
│   │   │
│   │   ├── ModelPerformanceVisualizer.md
│   │       # ✅ Updated visualization docs
│   │
│   ├── advanced/
│       ├── ComparisonPlots.py
│       │   # ✅ Baseline vs tuned comparison
│       │
│       ├── HyperparameterPlots.py
│       │   # ✅ Grid / Random search visualization
│       │
│       ├── OptimizationPlots.py
│           # ✅ Optimization trends & performance curves
```

---

## ✅ Core Layers

| Layer        | Responsibility         |
| ------------ | ---------------------- |
| Utility      | Orchestration          |
| Registry     | Model discovery        |
| Wrapper      | Pipeline + model       |
| Preprocessor | Feature transformation |
| Metrics      | Evaluation             |
| Tuner        | Optimization           |

---

## ✅ Key Improvements

- ✅ Wrapper-driven architecture
- ✅ Regression vs classification separation
- ✅ Fixed scoring bug in tuning
- ✅ Artifact-aware design
- ✅ Experiment-level tracking

---

## 🚀 End-to-End Flow

```python
lm = LinearModelUtility(df, "target")
lm.prepare_data()
lm.run_all_models()
lm.tune_model("Ridge", param_grid)
results = lm.get_results_df()
```

---

## ✅ Classification Flow

```
ClassificationModelUtility → Metrics.classification → ROC / CM
```

## ✅ Regression Flow

```
LinearModelUtility → Metrics.regression → R2 / MSE
```

---

## ✅ Key Fixes Implemented

- ❌ Removed misuse of classification scoring in regression
- ✅ Enforced proper scoring in tuner
- ✅ Fixed empty tuning results issue

---

## ✅ Design Principles

- ✅ Separation of concerns
- ✅ Loose coupling
- ✅ Extensibility
- ✅ Production-ready

---

## ✅ Future Scope

- AutoML orchestrator
- Multi-metric tuning
- Explainability integration

---

## ✅ Summary

This framework is now:

🚀 Production-ready
🚀 AutoML-compatible
🚀 Fully modular
🚀 Architecturally clean
