# ML HR Employee Pipeline

## Script

- `Module-3/ML/projects/ml_hr_employee.py`

## Overview

This pipeline builds an end-to-end employee attrition analysis workflow:

- Data quality assessment
- Unsupervised clustering on employees who left
- Classification modeling for turnover prediction
- Retention risk scoring and recommendation generation
- HTML report generation with charts and summary tables

The implementation is aligned with the shared framework utilities in `lib/utility/machinelearning`.

## Input Dataset

- `datasets/HR_comma_sep.csv`

## Workflow Sections

1. **Data Quality**

- DataFrame info
- Missing-value report
- Duplicate summary

2. **Unsupervised Clustering (Left Employees)**

- Uses `UnsupervisedModelUtility`
- Runs KMeans and DBSCAN on:
  - `satisfaction_level`
  - `last_evaluation`
- Builds cluster behavior summary

3. **Classification Pipeline**

- Uses `ClassificationModelUtility`
- Models:
  - LogisticRegression
  - RandomForestClassifier
  - GradientBoosting
- Uses SMOTE imbalance handling
- Selects best model by `recall_weighted`

4. **Risk Scoring**

- Uses best classification model probabilities
- Assigns risk zones:
  - Safe Zone
  - Low Risk Zone
  - Medium Risk Zone
  - High Risk Zone
- Adds retention recommendations

5. **EDA Visualizations**

- Histograms
- Correlation matrix
- Pair plot

6. **Report Output**

- Consolidated HTML report with cards and charts

## Output

- Report file:
  - `Module-3/ML/projects/reports/ml_hr_employee_pipeline_report.html`

## How To Run

From workspace root:

```powershell
c:/IHFC/python-learnings/.venv/Scripts/python.exe -m Module-3.ML.projects.ml_hr_employee
```

Alternative:

```powershell
py -m Module-3.ML.projects.ml_hr_employee
```

## Notes

- The script expects framework dependencies from `requirements.txt` to be installed.
- If model visualization payloads are missing, check model run results and `log/run_*.log`.
