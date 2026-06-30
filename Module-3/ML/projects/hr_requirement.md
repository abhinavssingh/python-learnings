# Employee Turnover Analytics – Requirements Document

## 1. Project Overview

### Business Context
Portobello Tech has developed an intelligent employee turnover prediction initiative to help the HR Department proactively identify employees who are at risk of leaving the organization.

Employee turnover refers to the total number of employees who leave an organization over a period of time. Historical employee data containing employee satisfaction, performance evaluations, workload, tenure, promotions, department information, and salary information will be analyzed using Machine Learning techniques.

The objective of this project is to:
- Improve employee retention.
- Identify key drivers of employee attrition.
- Build predictive models for turnover.
- Segment employees based on behavioral patterns.
- Recommend targeted retention strategies.

---

## 2. Project Objectives

The ML solution must:

1. Perform data quality checks.
2. Conduct Exploratory Data Analysis (EDA).
3. Cluster employees who left the company.
4. Address class imbalance using SMOTE.
5. Train and evaluate classification models using 5-Fold Cross Validation.
6. Identify the best-performing model using appropriate evaluation metrics.
7. Generate retention recommendations based on employee turnover risk.

---

## 3. Dataset Information

### Source Dataset
https://www.kaggle.com/liujiaqi/hr-comma-sepcsv

### Target Variable

| Column | Description |
|----------|-------------|
| left | 0 = Employee stays, 1 = Employee left |

### Data Dictionary

| Column Name | Description |
|------------|-------------|
| satisfaction_level | Employee job satisfaction score |
| last_evaluation | Most recent performance evaluation score |
| number_project | Number of projects handled |
| average_montly_hours | Average monthly working hours |
| time_spend_company | Years spent in the company |
| Work_accident | 0 = No accident, 1 = Accident |
| left | Attrition indicator |
| promotion_last_5years | Promotion count in last five years |
| Department | Employee department |
| salary | Salary category |

---

# 4. Functional Requirements

## FR-1 Data Quality Assessment

### Objective
Validate the quality and completeness of the dataset.

### Tasks

#### FR-1.1 Missing Value Analysis
- Identify missing values.
- Report missing value counts.
- Calculate missing value percentages.

### Deliverables
- Missing value report.
- Data quality summary.

---

## FR-2 Exploratory Data Analysis (EDA)

### Objective
Understand key factors contributing to employee turnover.

### FR-2.1 Correlation Analysis

#### Tasks
- Select numerical features.
- Compute correlation matrix.
- Visualize correlation matrix using heatmap.

#### Deliverable
- Correlation heatmap.
- Business interpretation of strongest correlations.

---

### FR-2.2 Feature Distribution Analysis

#### Create distribution plots for:

1. satisfaction_level
2. last_evaluation
3. average_montly_hours

#### Deliverables
- Histogram plots.
- Distribution summaries.
- Business insights.

---

### FR-2.3 Project Count Analysis

#### Tasks
- Visualize employee project counts.
- Compare employees who stayed vs left.

#### Visualization
- Bar Chart

#### Inputs
- number_project
- left

#### Deliverables
- Project count distribution.
- Attrition insights by project workload.

---

## FR-3 Employee Clustering Analysis

### Objective
Identify meaningful employee segments among employees who left.

---

### FR-3.1 Data Selection

Required Columns:
- satisfaction_level
- last_evaluation
- left

Filter:

```text
left = 1
```

---

### FR-3.2 K-Means Clustering

#### Tasks
- Apply K-Means clustering.
- Create exactly 3 clusters.
- Assign cluster labels.

#### Inputs
- satisfaction_level
- last_evaluation

#### Deliverables
- Cluster assignments.
- Cluster summary table.
- Cluster visualization.

---

### FR-3.3 Cluster Interpretation

For each cluster:

- Average satisfaction.
- Average evaluation.
- Employee count.
- Behavioral pattern.

#### Deliverable
- Cluster interpretation report.

---

## FR-4 Class Imbalance Handling

### Objective
Address class imbalance in employee attrition.

---

### FR-4.1 Data Preprocessing

#### Tasks

Separate:

- Numeric columns
- Categorical columns

Apply:

```python
pd.get_dummies()
```

to categorical features.

Combine transformed categorical and numerical features.

#### Deliverable
- Model-ready dataset.

---

### FR-4.2 Train-Test Split

#### Requirements

- Train: 80%
- Test: 20%
- Stratified split
- random_state=123

#### Deliverable
- Training dataset
- Test dataset

---

### FR-4.3 SMOTE Upsampling

#### Tasks

Apply:

```python
SMOTE()
```

Only on training dataset.

#### Deliverables

- Original class distribution.
- Balanced class distribution.
- Comparison report.

---

## FR-5 Model Training and Evaluation

### Objective
Train multiple classification models.

### Cross Validation Requirement

- 5-Fold Cross Validation

---

### FR-5.1 Logistic Regression

#### Tasks
- Train Logistic Regression.
- Apply 5-Fold CV.
- Generate classification report.

#### Deliverables
- CV scores.
- Classification report.

---

### FR-5.2 Random Forest Classifier

#### Tasks
- Train Random Forest.
- Apply 5-Fold CV.
- Generate classification report.

#### Deliverables
- CV scores.
- Classification report.

---

### FR-5.3 Gradient Boosting Classifier

#### Tasks
- Train Gradient Boosting.
- Apply 5-Fold CV.
- Generate classification report.

#### Deliverables
- CV scores.
- Classification report.

---

## FR-6 Best Model Identification

### Objective
Identify the most effective turnover prediction model.

---

### FR-6.1 ROC/AUC Analysis

#### Tasks
- Calculate ROC-AUC for all models.
- Plot ROC curves.

#### Deliverables
- ROC Curve Visualization.
- AUC Comparison Table.

---

### FR-6.2 Confusion Matrix Analysis

#### Tasks
- Generate confusion matrix for all models.

#### Deliverables
- Confusion matrices.
- False Positive analysis.
- False Negative analysis.

---

### FR-6.3 Metric Justification

#### Requirement
Explain whether Recall or Precision should be prioritized.

#### Expected Discussion
Since employee turnover prediction is a retention problem:

- False negatives are costly.
- Missing a likely-to-leave employee affects business continuity.

Therefore:

**Recall should generally be prioritized over Precision**, while balancing overall business objectives.

#### Deliverable
- Metric selection rationale.

---

## FR-7 Retention Strategy Recommendation Engine

### Objective
Create actionable retention recommendations.

---

### FR-7.1 Probability Scoring

#### Tasks
- Use the best model.
- Predict turnover probabilities.
- Score employees in the test dataset.

#### Deliverable
- Turnover probability report.

---

### FR-7.2 Risk Segmentation

Employees must be categorized into four risk zones.

| Zone | Probability Range | Risk Level |
|--------|------------------|------------|
| Safe Zone | < 20% | Green |
| Low Risk Zone | 20% - 60% | Yellow |
| Medium Risk Zone | 60% - 90% | Orange |
| High Risk Zone | > 90% | Red |

---

### Retention Strategy Recommendations

#### Safe Zone (Green)
- Continue engagement programs.
- Recognize achievements.
- Offer career development opportunities.

#### Low Risk Zone (Yellow)
- Conduct periodic check-ins.
- Understand emerging concerns.
- Improve team engagement.

#### Medium Risk Zone (Orange)
- Manager intervention.
- Career planning discussions.
- Workload review.
- Compensation review.

#### High Risk Zone (Red)
- Immediate HR engagement.
- Retention interviews.
- Compensation adjustment assessment.
- Role redesign or internal mobility opportunities.

---

# 5. Success Criteria

The project will be considered successful when:

- Data quality validation is completed.
- EDA insights are documented.
- Employee clusters are identified and interpreted.
- Class imbalance is addressed using SMOTE.
- All three models are trained using 5-Fold CV.
- Best model is selected based on evaluation metrics.
- Employee turnover probabilities are generated.
- Employees are segmented into risk zones.
- Retention recommendations are provided.
