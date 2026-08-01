# Home Loan Default Prediction using Deep Learning

## Overview

This project implements an end-to-end Home Loan Default Prediction pipeline using a custom Deep Learning framework built on TensorFlow.

The solution includes:

- Automated dataset loading and optimization
- Feature engineering
- Missing value analysis
- Categorical encoding
- Feature scaling
- Class imbalance handling using SMOTE
- Multi-Layer Perceptron (MLP) neural network
- Model training and evaluation
- ROC-AUC analysis
- Training visualization
- Automated HTML report generation

## Dataset

Expected input file: `loan_data.csv`

Target column: `TARGET`

| Target | Meaning     |
| ------ | ----------- |
| 0      | Non-Default |
| 1      | Default     |

## Solution Architecture

loan_data.csv → Dataset Loading → Data Profiling → Feature Engineering → Preprocessing → SMOTE → TensorFlow MLP → Evaluation → HTML Report

## Feature Engineering

- Income Credit Ratio
- Annuity Income Ratio
- Credit Goods Gap
- Income Per Family Member

## Data Preprocessing

- Missing value handling
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- SMOTE oversampling

## Model Configuration

- Hidden Layers: 256 → 128 → 64
- Activation: ReLU
- Dropout: 0.3
- Output: Sigmoid
- Optimizer: Adam
- Loss: Binary Crossentropy

## Training

- Epochs: 20
- Batch Size: 256
- Early Stopping Enabled
- Learning Rate Reduction Enabled

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Sensitivity
- ROC-AUC

## Generated Outputs

- Class Distribution Charts
- ROC Curve
- Training History Plots
- HTML Report

## Execution

```bash
python home_loan.py
```

## Report Output

```text
reports/home_loan_dl_report.html
```

## Future Enhancements

- Hyperparameter tuning
- Ensemble Learning
- SHAP Explainability
- MLflow Integration
- Model Deployment
