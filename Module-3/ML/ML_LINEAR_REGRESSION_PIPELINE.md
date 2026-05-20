# ML Linear Regression Pipeline Report Generator

This script builds a **complete machine learning pipeline dashboard** and exports it as an interactive HTML report.

It integrates:

- Data loading & cleaning
- Feature engineering
- Machine learning training
- Performance visualization
- HTML dashboard generation

---

## 🚀 Features

- 📊 End-to-end ML pipeline automation
- 🧹 Data preprocessing (imputation + outlier handling)
- 🧠 Multiple regression models training
- 📈 Interactive Plotly visualizations
- 🖥️ HTML dashboard generation (auto-open in browser)
- 🧩 Modular architecture using reusable utilities

---

## 📦 Dependencies

Ensure the following modules exist in your project:

- HtmlBuilder
- PlotRenderer
- DataLoader
- DataFrameHelper
- CustomImputer
- LinearModelUtility
- ModelPerformanceVisualizer
- OutlierHandler
- ReportUtils

External libraries:

```bash
pip install pandas plotly scikit-learn
```

---

## 🧠 Workflow

### 1. Load Dataset

```python
df, report = dl.read_dataset("marketing_data.csv", return_report=True)
```

---

### 2. Data Cleaning

- Convert `Income` to numeric
- Convert `Dt_Customer` to datetime

```python
df['Income'] = df['Income'].replace('[\$,]', '', regex=True).astype(float)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
```

---

### 3. Feature Engineering

Create total spend feature:

```python
Total_Mnt = df.loc[:, df.columns.str.contains('Mnt')].sum(axis=1)
```

---

### 4. Train Machine Learning Models

```python
ml = lmu(df, target_col="TotalSpend")

ml_results = ml.train_all(
    imputer=CustomImputer(),
    outlier_handler=OutlierHandler()
)
```

---

### 5. Visualization

Uses `ModelPerformanceVisualizer`:

- Model comparison
- Total error comparison
- Actual vs predicted plots

---

### 6. Build HTML Dashboard

```python
html_doc = builder.build_page(title, content)
```

---

### 7. Save Report

```python
ru.save_html_report(...)
```

- Saves file
- Opens automatically in browser

---

## 📊 Visualizations Included

- ✅ Model comparison (Ridge vs Lasso)
- ✅ All model comparison
- ✅ Total error comparison (all models)
- ✅ Actual vs predicted plots (all models)
- ✅ Total error per model

---

## 📂 Output

- File: `ml_linear_regression_pipeline_report.html`
- Location: `/reports` folder
- Opens automatically in browser

---

## ✅ Key Benefits

- End-to-end ML automation
- Interactive dashboards
- Clean architecture
- Plug-and-play components
- Portfolio-ready project

---

## 🔮 Possible Enhancements

- Add K-Fold visualization support
- Model ranking leaderboard
- Feature importance plots
- Deploy with Streamlit

---

## 📜 License

Free to use for learning and internal projects 🚀
