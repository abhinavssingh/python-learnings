# Python Learnings + Modular ML/DL Framework

This repository combines Python learning modules with reusable, utility-driven ML and deep learning frameworks.

It includes:

- Python fundamentals (NumPy, Pandas, core concepts)
- Utility-driven execution system
- Config-driven path management
- Classical ML pipeline (training -> evaluation -> visualization -> reporting)
- Deep Learning framework (TensorFlow model wrappers, data loaders, training utilities, and project pipelines)

---

## Overview

The project is divided into three areas:

### 1. Learning Modules

- Python basics (Module-1)
- NumPy / Pandas exercises (Module-2)
- Script-based reports and experiments

### 2. Machine Learning Framework

Located under `machinelearning/`, this includes:

- Wrapper-based model execution
- Pipeline-first design (Preprocessor → Model)
- Imbalance handling (SMOTE, SMOTEENN)
- Evaluation (metrics + artifacts)
- Comparator (ranking, best model selection)
- Visualization (VisualizerEngine)
- HTML report generation

### 3. Deep Learning Framework

Located under `lib/utility/deeplearning/` and `lib/utility/deeplearning/frameworks/tensorflow/`, this includes:

- Framework abstractions and configuration
- TensorFlow model wrappers (Dense, CNN, Sequence, Autoencoder)
- Data loaders for tabular, image, text, and autoencoder NPZ flows
- Training utilities and callback integration
- Visualization helpers (ROC, confusion matrix, reconstruction grids, training history)
- Hybrid ensemble support (LogisticRegression + RandomForest + SVC + Keras classifier via soft voting)

---

## Folder Structure

```text
.
│   .editorconfig        # Defines coding style across editors
│   .gitignore           # Files ignored by Git
│   settings.json        # Config-driven path resolution
│
├── .vscode/
│   └── tasks.json       # VS Code task runner configurations
│
├── assets/
│   └── input.css        # Tailwind input CSS for reports
│
├── log/                 # Runtime logs
│
├── lib/
│   ├── init.py          # Dynamic path resolver (core execution enabler)
│   ├── run.py           # Script runner (CLI execution engine)
│   ├── runlist.json     # Config-based execution sequence
│   ├── logger.py        # Logging utility (info/debug/error)
│   ├── report_utils.py  # Report generation helpers
│   └── html/            # HTML rendering engine
│
├── Module-1/            # Python fundamentals
│
├── Module-2/
│   ├── NumPy/           # NumPy learning scripts
│   └── Pandas/          # Pandas scripts and reports
├── Module-3/
│   └── ML/              # machine learning scripts
│
├── Module-4/
│   └── DeepML/          # deep learning projects and requirements
│
├── machinelearning/     # Core ML framework
│   ├── base/            # Wrappers + execution layer
│   ├── pipeline/        # Preprocessing + imbalance handling
│   ├── registry/        # Model discovery engine
│   ├── facade/          # Utility orchestration layer
│   ├── evaluation/      # Metrics + comparator layer
│   ├── shared/          # ResultBuilder + formatting
│   ├── tuning/          # Hyperparameter tuning
│   ├── visualization/   # VisualizerEngine + plots
│   └── reports/         # HTML reporting
│
└── lib/
   └── utility/
      ├── dataframe/   # dataframe loaders and helpers
      └── deeplearning/
         ├── abstractions/
         ├── config/
         ├── evaluation/
         ├── frameworks/
         │   └── tensorflow/
         │       ├── data/
         │       ├── models/
         │       ├── training/
         │       ├── pipelines/
         │       └── ensemble/
         ├── optimization/
         ├── preprocessing/
         └── visualization/


```

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/abhinavssingh/python-learnings.git
   ```
2. Navigate to the repo:
   ```bash
   cd python-learnings
   ```
3. Set up a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Optional: exclude notebook outputs from version control:
   ```gitignore
   **/.ipynb_checkpoints/
   *.ipynb
   ```
   Optional tool:
   ```bash
   pip install nbstripout
   nbstripout --install
   ```

---

## Script Execution

All scripts support three execution modes, enabled through `init.py`.

### 1. Direct Script Execution

```
python Module-2/NumPy/numpy_basics_report.py
```

This mode runs the script using its file location as the working directory.
`init.py` ensures imports such as:

```python
from lib.logger import log_info
from lib.html.base import build_html_page
```

work correctly.

---

### 2. Module Execution (`python -m package.module`)

```
python -m Module-2.NumPy.numpy_basics_report
```

This executes the script as a Python module.

---

### 3. Runner-Based Execution (`run.py`)

#### List all discovered modules:

```
python run.py --list
```

#### Run selected modules (pattern matching):

```
python run.py --only "Module-2.NumPy.*"
```

#### Run scripts defined in `runlist.json`:

```
python run.py --config runlist.json
```

---

## How Dynamic Path Resolution Works

### `init.py`

- Locates the project root by finding `settings.json`.
- Adds all configured folders to `sys.path`.
- Allows scripts to run without Python packages or `__init__.py` files.
- Ensures compatibility with all execution modes.

### `settings.json`

Defines importable directories:

```
{
    "paths": [
        "lib",
        "lib/tools",
        "Module-1",
        "Module-2"
    ]
}
```

## Code Formatting

Format all Python files recursively:

```bash
autopep8 --in-place --recursive .
```

---

## Machine Learning Framework

Located in `machinelearning/`.

### Key Features

- Wrapper-based execution (classification, regression, unsupervised)
- Pipeline-first architecture
- Imbalance handling (SMOTE, SMOTEENN)
- Ensemble support (Parallel, Sequential, Stacking)
- Hyperparameter tuning
- Artifact-aware evaluation
- Visualization via VisualizerEngine
- HTML report generation

### ML Execution Flow

```
prepare_data()
↓
Preprocessor
↓
Wrapper Pipeline
↓
(Optional) SMOTE (training only)
↓
Model / Ensemble / fit_predict
↓
Metrics + Artifacts
↓
ResultBuilder
↓
Comparator (ranking)
↓
VisualizerEngine
↓
HTML Report
```

### Visualization Architecture

```
Results + Artifacts
        ↓
VisualizerEngine ✅
        ↓
├── ComparisonPlots
├── DistributionPlots
├── Classification / Regression / Clustering
└── Dimensionality (PCA / t-SNE)
```

## Deep Learning Framework

Located in `lib/utility/deeplearning/`.

### Key Features

- TensorFlow wrappers for dense, CNN, sequence, and autoencoder models
- Reusable data loaders:
  - Image classification (`tf_image_classification_data_loader.py`)
  - Text classification (`tf_text_classification_data_loader.py`)
  - Denoising autoencoder NPZ loader (`tf_autoencoder_data_loader.py`)
- Reusable visualizations:
  - `training_history_plot.py`
  - `roc_curve_plot.py`
  - `confusion_matrix_plot.py`
  - `reconstruction_plot.py`
- Hybrid soft-voting ensemble utilities:
  - `tf_keras_classifier.py`
  - `tf_voting_classifier_factory.py`

### DeepML Project Examples

- Face mask transfer learning
- Home loan default prediction
- Lending club default prediction
- Product review classification (CNN-LSTM)
- Dental x-ray denoising autoencoder
- Voting classifier demo (LR + RF + SVC + Keras)

## Tests

```Shell
python test.py
```

Validates:

- direct execution
- module execution
- runner discovery
- pattern matching
- config-based execution

---

## Tech Stack

- Python 3.x
- NumPy / Pandas
- Scikit-learn
- TensorFlow / Keras
- Plotly
- Tailwind CSS
- VS Code
- Git

---

## Design Principles

- Modular architecture
- Loose coupling (wrapper + registry pattern)
- Pipeline-first design
- Experiment-driven workflows
- Artifact-aware evaluation
- Config-driven extensibility
- Production-ready design

---

## Future Scope

- AutoML orchestration
- Multi-metric optimization
- Explainability (SHAP / LIME)
- Interactive dashboards
- Model monitoring

---

## Contributing

**Contributions** are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.
