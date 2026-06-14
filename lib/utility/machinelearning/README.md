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
│   # ✅ High-level overview of ML framework, architecture, usage
│
├── base/
│   ├── MLModelBase.py
│   │   # ✅ Abstract base class defining common ML interface (train, predict, pipeline)
│   │
│   ├── BaseModelWrapper.py
│   │   # ✅ Generic wrapper for pipeline + model integration
│   │
│   ├── ClassificationModelWrapper.py
│   │   # ✅ Wrapper specialized for classification models
│   │
│   ├── LinearRegressionModelWrapper.py
│       # ✅ Wrapper specialized for regression models
│
├── evaluation/
│   ├── Metrics.py
│   │   # ✅ Core computation engine (accuracy, f1, roc, etc.)
│   │
│   ├── METRICS.md
│   │   # ✅ Documentation for Metrics logic and usage
│   │
│   ├── ModelComparator.py
│   │   # ✅ Generic comparator for ranking models (shared logic)
│   │
│   ├── ClassificationModelComparator.py
│       # ✅ Classification-specific ranking, best model selection
│
├── facade/
│   ├── ClassificationModelUtility.py
│   │   # ✅ Main orchestration layer (classification pipeline: train, eval, tuning)
│   │
│   ├── CLASSIFICATIONMODELUTILITY.md
│   │   # ✅ Detailed documentation (flow, usage, architecture)
│   │
│   ├── LinearModelUtility.py
│   │   # ✅ Orchestration layer for regression workflows
│   │
│   ├── LinearModelUtility.md
│       # ✅ Documentation for regression utility
│
├── models/
│   ├── __init__.py
│   │   # ✅ Module initialization
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   │   # ✅ Registers classification models
│   │   │
│   │   ├── LogisticRegressionWrapper.py
│   │   │   # ✅ Wrapper for Logistic Regression
│   │   │
│   │   ├── DecisionTreeClassifierWrapper.py
│   │   │   # ✅ Wrapper for Decision Tree
│   │   │
│   │   ├── RandomForestWrapper.py
│   │   │   # ✅ Wrapper for Random Forest
│   │   │
│   │   ├── KNNClassifierWrapper.py
│   │   │   # ✅ Wrapper for KNN
│   │   │
│   │   ├── SVCWrapper.py
│   │       # ✅ Wrapper for Support Vector Classifier
│   │
│   ├── linear/
│       ├── __init__.py
│       │   # ✅ Registers regression models
│       │
│       ├── LinearRegressionWrapper.py
│       │   # ✅ Linear Regression model wrapper
│       │
│       ├── RidgeWrapper.py
│       │   # ✅ Ridge model wrapper
│       │
│       ├── LassoWrapper.py
│       │   # ✅ Lasso model wrapper
│       │
│       ├── ElasticNetWrapper.py
│           # ✅ ElasticNet model wrapper
│
├── pipeline/
│   ├── Preprocessor.py
│   │   # ✅ Builds full preprocessing pipeline (encoding, scaling, etc.)
│   │
│   ├── CustomImputer.py
│   │   # ✅ Handles missing values with advanced grouped strategies
│   │
│   ├── CustomImputer.md
│   │   # ✅ Documentation for imputation logic
│   │
│   ├── OutlierHandler.py
│   │   # ✅ Handles outliers (IQR, z-score, etc.)
│   │
│   ├── OutlierHandler.md
│       # ✅ Documentation for outlier handling
│
├── registry/
│   ├── ModelRegistry.py
│       # ✅ Central registry for model discovery & dynamic loading
│
├── shared/
│   ├── DataCleaner.py
│   │   # ✅ Cleans DataFrame (NaN handling, filtering for plots/metrics)
│   │
│   ├── Formatter.py
│   │   # ✅ Generic formatter interface (base formatting utilities)
│   │
│   ├── ClassificationFormatter.py
│       # ✅ Formats classification artifacts (CM, ROC, PR → DataFrame/UI)
│
├── tuning/
│   ├── HyperparameterTuner.py
│   │   # ✅ Generic hyperparameter tuning engine (grid/random)
│   │
│   ├── HyperparameterTuner.md
│   │   # ✅ Documentation for tuning strategies
│   │
│   ├── ClassificationHyperparameterTuner.py
│       # ✅ Classification-specific tuning logic and evaluation hooks
│
├── visualization/
│   ├── README.md
│   │   # ✅ Visualization module overview and usage
│   │
│   ├── core/
│   │   ├── MetricResolver.py
│   │       # ✅ Dynamically resolves best metrics for plotting
│   │
│   ├── generic/
│   │   ├── ClassificationPlots.py
│   │   │   # ✅ Core visualization (bar, scatter, ROC, multi-metric)
│   │   │
│   │   ├── CLASSIFICATIONPLOTS.md
│   │   │   # ✅ Documentation for classification visualizations
│   │   │
│   │   ├── ModelPerformanceVisualizer.py
│   │   │   # ✅ Generic visualization wrapper for results
│   │   │
│   │   ├── ModelPerformanceVisualizer.md
│   │       # ✅ Documentation for visualizer layer
│   │
│   ├── advanced/
│       ├── ComparisonPlots.py
│       │   # ✅ Advanced model comparison visualizations
│       │
│       ├── HyperparameterPlots.py
│       │   # ✅ Visualize tuning results (grid/random search)
│       │
│       ├── OptimizationPlots.py
│           # ✅ Visualize optimization trends & performance curves
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
