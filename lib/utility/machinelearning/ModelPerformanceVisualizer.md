# ModelPerformanceVisualizer

A reusable Plotly-based visualization utility for evaluating machine learning model performance. Designed to integrate seamlessly with `LinearModelUtility` outputs.

---

## 🚀 Features

- 📊 Compare all models automatically
- 🎯 Compare selected models
- 📉 Actual vs Predicted visualization
- 📏 Residual diagnostics
- 🔍 Total error visualization (absolute & squared)
- 🧩 Flat JSON support (no nested parsing required)

---

## 🧠 Design Philosophy

- Uses flattened results (`get_flat_result`) for simplicity
- Separates computation and visualization concerns
- Handles missing data safely (e.g., skips models without predictions)

---

## 📦 Dependencies

```bash
pip install pandas plotly
```

---

## 📥 Usage

```python
viz = ModelPerformanceVisualizer(results)

# Compare all models
viz.plot_all_model_comparison().show()

# Selected models
viz.plot_model_comparison(["Ridge", "Lasso"]).show()

# Individual model plots
viz.plot_actual_vs_predicted("Ridge").show()
viz.plot_residuals("Ridge").show()

# Total error
viz.plot_total_error("Ridge", mode="absolute").show()
viz.plot_total_error_all(mode="squared").show()
```

---

## 📊 Available Plots

### 1. All Model Comparison

- Compares all models using R² and MSE
- Dual-axis visualization

### 2. Selected Model Comparison

- Compare only specific models

### 3. Actual vs Predicted

- Scatter plot
- Includes ideal reference line

### 4. Residual Plot

- Residual = y_true - y_pred
- Helps detect bias and patterns

### 5. Total Error (Single Model)

Modes:

- `absolute` → Σ |y_true - y_pred|
- `squared` → Σ (y_true - y_pred)²

### 6. Total Error (All Models)

- Compares total error across models
- Skips models without predictions

---

## 🔧 Helper Function

### get_flat_result

Flattens nested result dictionary for easy access:

```python
flat = viz.get_flat_result("Ridge")
print(flat["MSE"], flat["y_true"])
```

---

## ✅ Key Benefits

- No nested dictionary handling
- Ready for dashboards (Plotly / Streamlit)
- Clean and reusable design
- Robust to mixed training modes

---

## 📜 License

Free to use for learning and projects 🚀
