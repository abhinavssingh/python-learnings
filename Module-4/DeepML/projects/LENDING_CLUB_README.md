# Lending Club Loan Default Prediction using Deep Learning

## Overview

This project builds an end-to-end loan default prediction system using TensorFlow-based Deep Learning. The solution analyzes Lending Club loan data and predicts whether a loan will be fully repaid or default.

## Key Features

- Automated dataset profiling
- Feature engineering
- Correlation analysis
- Missing value analysis
- Categorical encoding
- Feature scaling
- SMOTE class balancing
- TensorFlow MLP model
- ROC-AUC evaluation
- Confusion matrix analysis
- Interactive HTML reporting

## Dataset

Input dataset:

```text
loan_data (2).csv
```

Target column:

```text
not.fully.paid
```

| Value | Meaning        |
| ----- | -------------- |
| 0     | Fully Paid     |
| 1     | Not Fully Paid |

## Feature Engineering

Generated features:

- annual.inc
- installment_income_ratio
- revolbal_income_ratio

## Data Preparation

- Missing value assessment
- Duplicate detection
- High correlation removal
- Label Encoding
- One-Hot Encoding
- Standard Scaling
- SMOTE Oversampling

## Neural Network Architecture

```text
Input Layer
    ↓
Dense(128) ReLU
    ↓
Dropout(0.25)
    ↓
Dense(64) ReLU
    ↓
Dropout(0.25)
    ↓
Dense(32) ReLU
    ↓
Dense(1) Sigmoid
```

## Training Configuration

- Optimizer: Adam
- Learning Rate: 0.001
- Loss: Binary Crossentropy
- Epochs: 40
- Batch Size: 128
- Early Stopping Enabled
- Learning Rate Reduction Enabled

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- True Positives / Negatives
- False Positives / Negatives

## Generated Visualizations

- Class Distribution
- Loan Purpose Distribution
- Default Rate by Purpose
- Interest Rate Distribution
- FICO Distribution
- Correlation Heatmap
- ROC Curve
- Training History
- Confusion Matrix

## Output Report

```text
reports/lending_club_dl_report.html
```

## Run

```bash
python lending_club.py
```

## Project Structure

```text
project/
├── lending_club.py
├── loan_data (2).csv
├── reports/
│   └── lending_club_dl_report.html
└── README.md
```
