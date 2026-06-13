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

### 🔷 High-Level Flow

```mermaid
flowchart TD
    A[User API - Utility] --> B[ModelRegistry]
    A --> C[Preprocessor]
    A --> D[ExperimentRunner]
    D --> E[Model Wrapper]
    E --> F[Pipeline]
    F --> G[Predictions]
    G --> H[Metrics]
    H --> I[Results]
```

---

### 🔁 Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant LU as LinearModelUtility
    participant MR as ModelRegistry
    participant ER as ExperimentRunner
    participant W as Wrapper

    U->>LU: run_experiment()
    LU->>MR: get_model()
    MR-->>LU: wrapper
    LU->>W: build_pipeline()
    LU->>ER: run()
    ER->>W: train()
    ER->>W: predict()
    ER->>W: evaluate()
    ER-->>LU: results
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
├── experiment/
│   ├── ExperimentRunner.py
│       # ✅ Executes training pipelines, manages experiment lifecycle
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

## 🧩 Layer-by-Layer Explanation

### 1. Base Layer (`base/`)

#### MLModelBase

Defines a contract for:

- Data preparation
- Experiment execution
- Model tuning
- Evaluation and comparison

#### BaseModelWrapper

Encapsulates model logic:

- Pipeline building
- Training and prediction
- Model-specific evaluation

```python
wrapper.build_pipeline(preprocessor)
wrapper.train(X_train, y_train)
preds = wrapper.predict(X_test)
metrics = wrapper.evaluate(y_test, preds)
```

---

### 2. Model Layer (`models/`)

Each model has its own wrapper class.

Example:

- LinearRegressionWrapper
- RidgeWrapper
- LassoWrapper
- ElasticNetWrapper

Responsibilities:

- Build pipeline
- Define evaluation metrics

```python
class RidgeWrapper(BaseModelWrapper):
    def build_pipeline(self, preprocessor):
        ...
```

---

### 3. Registry Layer (`registry/`)

#### ModelRegistry

- Stores all available models
- Returns model wrappers
- Enables plug-and-play architecture

```python
registry = ModelRegistry()
model = registry.get_model("Ridge")
```

---

### 4. Pipeline Layer (`pipeline/`)

#### Preprocessor

Builds preprocessing pipeline:

- Numeric scaling
- Categorical encoding
- Optional imputation
- Optional outlier handling

```python
preprocessor = Preprocessor(X, imputer, outlier).build()
```

---

### 5. Experiment Layer (`experiment/`)

#### ExperimentRunner

Handles:

- Training
- Prediction
- Evaluation
- Experiment result tracking

```python
runner.run(model_name, wrapper, X_train, X_test, y_train, y_test)
```

---

### 6. Tuning Layer (`tuning/`)

#### HyperparameterTuner

Supports:

- Grid Search
- Random Search

```python
result = tuner.grid_search(pipeline, param_grid)
```

---

### 7. Evaluation Layer (`evaluation/`)

#### Metrics

Provides:

- Regression metrics (R2, MSE, RMSE)
- Classification metrics (Accuracy, F1)

```python
Metrics.regression(y_true, y_pred)
```

#### ModelComparator

Supports:

- Model ranking
- Best model selection
- Baseline vs tuned comparison

---

### 8. Facade Layer (`facade/`)

#### LinearModelUtility

Main user-facing class:

- Prepare data
- Run experiments
- Perform tuning
- Compare models

#### ClassificationModelUtility

## Similar interface for classification tasks.

## 🧪 Example Usage (End-to-End)

```python
lm = LinearModelUtility(df, target_col="target")

lm.prepare_data()

lm.run_experiment("Ridge")

lm.run_all_models()

lm.grid_search_cv("Ridge", param_grid)

best = lm.get_best_model("R2")
```

---

## 📚 API Documentation

## LinearModelUtility

### prepare_data()

Prepares dataset and builds preprocessing pipeline.

### run_experiment(model_name, k_fold=None)

Runs a single experiment.

### run_all_models()

Runs all models.

### grid_search_cv()

Performs grid search.

---

## BaseModelWrapper

### build_pipeline()

Constructs ML pipeline.

### train()

Fits model.

### predict()

Generates predictions.

---

## ModelRegistry

### get_model(name)

Returns model wrapper.

---

## 🔄 Workflow

1. Initialize utility
2. Prepare data
3. Build preprocessing pipeline
4. Fetch model from registry
5. Run experiment using runner
6. Evaluate and store results

---

## ✅ Key Benefits

- Loose coupling between components
- Highly extensible (add new models easily)
- Reusable pipelines and experiments
- Clean separation of responsibilities
- Production-ready architecture

---

## 🚀 How to Extend

### Add new model

1. Create wrapper in `models/`
2. Register in `ModelRegistry`

### Add new metric

1. Add method in `Metrics.py`

### Add new preprocessing step

1. Modify `Preprocessor.py`

---

## 📌 Summary

This framework follows best practices:

- SOLID principles
- Modular design
- Scalable architecture

It is suitable for:

- Production ML systems
- Rapid experimentation
- Extensible AutoML systems

---
