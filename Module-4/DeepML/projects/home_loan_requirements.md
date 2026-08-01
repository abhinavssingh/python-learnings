# Home Loan Data Analysis - Requirements

## Project Title
Home Loan Default Prediction Using Deep Learning (Keras & TensorFlow)

## Domain
Finance

## Problem Statement
For a safe and secure lending experience, analyze historical loan data and build a deep learning model capable of predicting the probability of loan default for future applicants.

The dataset is highly imbalanced and contains a large number of features, making the prediction task challenging.

## Objective
Create a deep learning model that predicts whether an applicant will be able to repay a loan using historical loan data.

## Dataset Analysis Requirements

### 1. Load Dataset
- Load the provided dataset.
- Verify structure, dimensions and data quality.

### 2. Data Quality Assessment
- Check for null values.
- Identify missing data percentages.
- Review data types.
- Detect potential inconsistencies.

### 3. Target Variable Analysis
- Analyze the TARGET column.
- Calculate the percentage of:
  - Default cases
  - Non-default cases
- Determine whether the dataset is balanced or imbalanced.

### 4. Data Balancing
If the dataset is imbalanced:
- Apply appropriate balancing techniques.
- Possible approaches:
  - Random Oversampling
  - Random Undersampling
  - SMOTE
  - Class Weights

### 5. Data Visualization
Create visualizations for:
- Original class distribution
- Balanced class distribution
- Target variable analysis

### 6. Feature Engineering
- Identify categorical columns.
- Identify numerical columns.
- Encode columns required by the model.
- Apply preprocessing and transformations.

### 7. Data Preparation
- Prepare features (X).
- Prepare target variable (y).
- Split data into training and testing datasets.
- Scale features where required.

## Deep Learning Model Requirements

### Model Development
Build a Deep Learning model using:
- TensorFlow
- Keras

Possible architecture:
- Input Layer
- One or More Hidden Layers
- Output Layer

### Training
- Train on historical loan data.
- Validate model performance.
- Tune parameters where required.

## Evaluation Requirements

### Classification Metrics
Calculate:
- Accuracy
- Precision
- Recall
- F1 Score

### Sensitivity
Calculate Sensitivity (True Positive Rate):

Sensitivity = TP / (TP + FN)

This metric is mandatory.

### ROC Curve
Generate Receiver Operating Characteristic (ROC) Curve.

### AUC Score
Calculate:
- Area Under ROC Curve (AUC)

This metric is mandatory.

## Deliverables

### Data Analysis
- Missing value analysis
- Target distribution analysis
- Data balancing analysis

### Visualizations
- Imbalanced dataset plot
- Balanced dataset plot
- ROC Curve

### Model
- Deep Learning model implementation
- Model training results

### Evaluation
- Sensitivity
- ROC Curve
- AUC Score
- Classification Metrics

### Final Outcome
Predict whether a loan applicant is likely to default or successfully repay the loan.
