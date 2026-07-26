# Capstone Session 9 Requirements

## Project Title

Predicting Customer Churn with Artificial Neural Networks

## Business Objective

Build an Artificial Neural Network (ANN) to identify bank customers who are likely to leave the bank (churn) based on historical customer data.

Target Variable:

- Exited
  - 1 = Customer left the bank
  - 0 = Customer stayed with the bank

## Dataset

File:

- Churn_Modelling.csv

## Data Preparation

1. Load Churn_Modelling.csv
2. Drop columns:
   - RowNumber
   - CustomerId
   - Surname
3. Create:
   - X (independent variables)
   - y = Exited
4. Label Encode:
   - Gender
5. One Hot Encode:
   - Geography
6. Train Test Split:
   - 80:20
   - random_state=0

## ANN Architecture

Layer 1:

- Dense(units=6, activation='relu')

Output Layer:

- Dense(units=1, activation='sigmoid')

## Compilation

- Optimizer: Adam
- Loss: binary_crossentropy
- Metrics: accuracy

## Training

- Epochs: 10
- Batch Size: 10

## Evaluation

- Accuracy Score
- Confusion Matrix

## Task B Prediction

Predict churn for:

- Geography: France
- Credit Score: 600
- Gender: Male
- Age: 40
- Tenure: 3
- Balance: 60000
- NumOfProducts: 2
- HasCrCard: Yes
- IsActiveMember: Yes
- EstimatedSalary: 50000

Output:

- 0 = Stay
- 1 = Leave

## Deliverables

- Data Preparation
- Feature Engineering
- ANN Model
- Training
- Evaluation
- Confusion Matrix
- Customer Churn Prediction
- Final Report
