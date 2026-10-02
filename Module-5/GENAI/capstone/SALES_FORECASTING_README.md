# Restaurant Sales Forecasting

Loads the Capstone 3 restaurant, item, and sales CSV files; validates and joins
them; reports sales trends and high-performing restaurants/items; compares
Linear Regression, Random Forest, and XGBoost; and recursively forecasts daily
revenue by restaurant.

## Run

This project now lives in the capstone folder. From the repository root, run:

```powershell
.\.venv\Scripts\python.exe Module-5\GENAI\capstone\sales_forecasting.py
```

The default inputs are in `datasets/GEN AI/Capstone 3/`:
`resturants.csv` (spelling as supplied), `items.csv`, and `sales.csv`.
Override the folder with `GENAI_SALES_DATA_DIR`.

By default, the script uses the final 28 days as a recursive validation
window, then forecasts the next 365 days. Configure these with
`GENAI_SALES_VALIDATION_DAYS` and `GENAI_SALES_HORIZON_DAYS`.

The model-validation split is time ordered. Features include calendar
indicators, seasonal terms, revenue lags, and rolling means; future predictions
are fed back into later lag features. Revenue is defined as transaction
`price * item_count`. Forecasts are daily per restaurant; the report includes
monthly, quarterly, yearly, restaurant, weekday, and item-level summaries.

The supplied item data does not contain a category column, so category-level
analysis is identified as unavailable instead of assigning guessed categories.
This is a classical forecasting project rather than an LLM task. It uses the
GenAI utility's `GenAIConfig`, `TabularLoader`, and `GenAIReport` for shared
configuration, data loading, and HTML reporting; it does not call Ollama.

## Outputs

- `Module-5/GENAI/capstone/reports/sales_forecasting_report.html`
- `saved_models/genai/sales_forecasting/model_metrics.csv`
- `saved_models/genai/sales_forecasting/daily_sales.csv`
- `saved_models/genai/sales_forecasting/future_sales_forecast.csv`
- `saved_models/genai/sales_forecasting/best_sales_forecaster.joblib`

Set `GENAI_OPEN_REPORTS=0` to save the HTML report without opening a browser.
Dependencies are covered by the repository's root requirements, including
Pandas, NumPy, scikit-learn, Plotly, and XGBoost.
