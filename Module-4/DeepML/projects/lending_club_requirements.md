# Lending Club Loan Data Analysis - Requirements

## Project Title
Lending Club Loan Default Prediction Using Deep Learning

## Domain
Finance

## Objective
Create a model that predicts whether a loan will default using historical Lending Club loan data.

## Problem Statement
Using Lending Club historical loan data from 2007-2015, build a deep learning model capable of predicting loan defaults. The dataset is highly imbalanced and contains numerous financial features that require preprocessing, feature engineering, exploratory analysis, and predictive modeling.

## Dataset Features

### Target Variable
- not.fully.paid (loan default indicator assumed as prediction target)

### Available Features
- credit.policy
- purpose
- int.rate
- installment
- log.annual.inc
- dti
- fico
- days.with.cr.line
- revol.bal
- revol.util
- inq.last.6mths
- delinq.2yrs
- pub.rec

## Project Requirements

### 1. Data Loading
- Load Lending Club dataset.
- Validate schema and quality.
- Review shape and data types.

### 2. Data Quality Assessment
- Check missing values.
- Analyze duplicate records.
- Review feature distributions.
- Detect anomalies and outliers.

### 3. Feature Transformation
- Transform categorical features into numerical values.
- Encode purpose and other categorical columns.
- Prepare dataset for deep learning.

### 4. Exploratory Data Analysis (EDA)
Perform detailed analysis of:
- Loan purpose
- Interest rate distributions
- Credit policy impact
- Debt-to-income ratio
- FICO score distribution
- Revolving balance and utilization
- Delinquency history
- Public records
- Default rate patterns

### 5. Feature Engineering
- Create derived features where appropriate.
- Identify highly correlated features.
- Generate correlation matrix.
- Remove highly correlated features.
- Retain most predictive features.

### 6. Class Imbalance Analysis
- Measure class distribution.
- Analyze default vs non-default ratio.
- Apply balancing strategy if needed.

Possible approaches:
- SMOTE
- Oversampling
- Undersampling
- Class Weights

### 7. Dataset Preparation
- Prepare feature matrix (X).
- Prepare target variable (y).
- Perform train/test split.
- Scale numerical features.

## Deep Learning Model

### Framework
- TensorFlow
- Keras

### Recommended Architecture
- Input Layer
- Hidden Layers
- Output Layer

### Output Layer
- Binary Classification
- Sigmoid Activation

### Compilation
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy

### Training
- Train on historical loan data.
- Monitor validation performance.
- Apply callbacks if required.

## Evaluation Requirements

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Confusion Matrix
- TP
- TN
- FP
- FN

### ROC Analysis
- ROC Curve
- AUC Score

## Deliverables

### Data Analysis
- Missing Value Analysis
- EDA Findings
- Correlation Analysis

### Feature Engineering
- Encoded Features
- Correlation Reduction
- Selected Features

### Deep Learning Solution
- TensorFlow/Keras Model
- Training Results
- Validation Results

### Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC
- Confusion Matrix

### Final Outcome
Predict whether a Lending Club loan applicant is likely to default using historical financial and credit-related information.
