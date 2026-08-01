# Bank Churn Prediction using Artificial Neural Network (ANN)

## Overview

This project implements a Bank Customer Churn Prediction solution using TensorFlow-based Artificial Neural Networks (ANN).

The model analyzes customer demographics, banking behavior, and account information to predict whether a customer is likely to leave the bank.

## Features

- Automated dataset loading
- Data preprocessing and feature engineering
- Label encoding and one-hot encoding
- Feature scaling
- ANN model training
- Model evaluation
- Single customer prediction
- Automated HTML report generation

## Dataset

Input file:

```text
Churn_Modeling.csv
```

Target column:

```text
Exited
```

| Value | Meaning         |
| ----- | --------------- |
| 0     | Customer Stays  |
| 1     | Customer Leaves |

## Preprocessing

Dropped Columns:

- RowNumber
- CustomerId
- Surname

Label Encoding:

- Gender

One Hot Encoding:

- Geography

Feature Scaling:

- Enabled for numerical attributes

## Model Architecture

```text
Input Layer
    ↓
Dense(6, ReLU)
    ↓
Dense(1, Sigmoid)
```

## Training Configuration

- Optimizer: Adam
- Learning Rate: 0.001
- Loss: Binary Crossentropy
- Epochs: 10
- Batch Size: 10

## Evaluation

The solution calculates:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Metrics

## Customer Prediction Example

The application performs churn prediction for a sample customer profile and returns:

- Predicted outcome (Stay/Leave)
- Churn probability

## Generated Report

```text
reports/bank_churn_ann_report.html
```

## Run

```bash
python bank_churn_ann.py
```

## Project Structure

```text
project/
├── bank_churn_ann.py
├── Churn_Modeling.csv
├── reports/
│   └── bank_churn_ann_report.html
└── README.md
```
