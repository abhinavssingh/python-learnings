"""Restaurant sales analysis and recursive daily revenue forecasting."""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.loaders import TabularLoader  # noqa: E402
from lib.utility.genai.reports import GenAIReport  # noqa: E402

DATA_FILES = {
    "restaurants": "resturants.csv",
    "items": "items.csv",
    "sales": "sales.csv",
}

LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
FEATURES = [
    "store_id",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "day_of_year",
    "is_weekend",
    "trend_days",
    "year_sin",
    "year_cos",
    *[f"lag_{lag}" for lag in LAGS],
    *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
]


def _read_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {key: data_dir / filename for key, filename in DATA_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required sales dataset file(s) not found: {', '.join(missing)}")
    return tuple(TabularLoader(paths[key]).read_frame() for key in ("restaurants", "items", "sales"))


def load_and_prepare(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load, validate, clean, and join transactions to item and restaurant details."""
    restaurants, items, sales = _read_csvs(data_dir)
    required = {
        "restaurants": (restaurants, {"id", "name"}),
        "items": (items, {"id", "store_id", "name"}),
        "sales": (sales, {"date", "item_id", "price", "item_count"}),
    }
    for label, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{label} dataset is missing required columns: {', '.join(sorted(missing))}")

    quality = {
        "restaurants_rows_loaded": len(restaurants),
        "items_rows_loaded": len(items),
        "sales_rows_loaded": len(sales),
        "sales_duplicate_rows_removed": int(sales.duplicated().sum()),
        "sales_rows_missing_values": int(sales.isna().any(axis=1).sum()),
    }
    restaurants = restaurants.drop_duplicates().drop_duplicates(subset=["id"], keep="first")
    items = items.drop_duplicates().drop_duplicates(subset=["id"], keep="first")
    sales = sales.drop_duplicates().copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    for column in ("item_id", "price", "item_count"):
        sales[column] = pd.to_numeric(sales[column], errors="coerce")

    invalid = (
        sales["date"].isna()
        | sales["item_id"].isna()
        | sales["price"].isna()
        | sales["item_count"].isna()
        | (sales["price"] < 0)
        | (sales["item_count"] < 0)
    )
    quality["invalid_sales_rows_removed"] = int(invalid.sum())
    sales = sales.loc[~invalid].copy()
    if sales.empty:
        raise ValueError("No valid sales rows remain after data quality checks.")

    sales["item_id"] = sales["item_id"].astype(int)
    items = items.rename(columns={"id": "item_id"})
    restaurants = restaurants.rename(columns={"id": "store_id", "name": "restaurant_name"})
    merged = sales.merge(items, on="item_id", how="left", validate="many_to_one", indicator="_item_merge")
    quality["sales_without_item_match"] = int(merged["_item_merge"].ne("both").sum())
    merged = merged.drop(columns="_item_merge")
    merged = merged.merge(restaurants, on="store_id", how="left", validate="many_to_one",
                          indicator="_restaurant_merge")
    quality["sales_without_restaurant_match"] = int(merged["_restaurant_merge"].ne("both").sum())
    merged = merged.loc[merged["_restaurant_merge"].eq("both")].drop(columns="_restaurant_merge")
    merged = merged.loc[merged["item_id"].notna() & merged["store_id"].notna()].copy()
    if merged.empty:
        raise ValueError("No sales rows matched both item and restaurant records.")

    merged["item_id"] = merged["item_id"].astype(int)
    merged["store_id"] = merged["store_id"].astype(int)
    merged["sales_revenue"] = merged["price"] * merged["item_count"]
    quality["sales_rows_after_cleaning"] = len(merged)
    return merged, restaurants, quality


def aggregate_daily_sales(transactions: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    """Create a complete daily revenue series per restaurant, filling closed/missing days with zero."""
    actual = (
        transactions.groupby(["date", "store_id"], as_index=False)
        .agg(sales_revenue=("sales_revenue", "sum"), units_sold=("item_count", "sum"))
    )
    dates = pd.date_range(actual["date"].min(), actual["date"].max(), freq="D")
    stores = restaurants[["store_id", "restaurant_name"]].drop_duplicates("store_id")
    complete = pd.MultiIndex.from_product(
        [dates, stores["store_id"].tolist()], names=["date", "store_id"]
    ).to_frame(index=False)
    daily = complete.merge(actual, on=["date", "store_id"], how="left")
    daily[["sales_revenue", "units_sold"]] = daily[["sales_revenue", "units_sold"]].fillna(0.0)
    return daily.merge(stores, on="store_id", how="left", validate="many_to_one")


def _calendar_features(date: pd.Timestamp, store_id: int, start_date: pd.Timestamp) -> dict[str, float]:
    day = date.dayofyear
    angle = 2 * np.pi * day / 365.25
    return {
        "store_id": store_id,
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "quarter": date.quarter,
        "day_of_year": day,
        "is_weekend": int(date.dayofweek >= 5),
        "trend_days": (date - start_date).days,
        "year_sin": float(np.sin(angle)),
        "year_cos": float(np.cos(angle)),
    }


def make_training_features(daily: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    """Create lagged revenue features without using same-day or future observations."""
    frame = daily.sort_values(["store_id", "date"]).copy()
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame.groupby("store_id")["sales_revenue"].shift(lag)
    prior_sales = frame.groupby("store_id")["sales_revenue"].shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = (
            prior_sales.groupby(frame["store_id"]).rolling(window, min_periods=window).mean()
            .reset_index(level=0, drop=True)
        )
    calendar = frame["date"].map(lambda date: _calendar_features(date, 0, start_date))
    for column in ("day_of_week", "day_of_month", "month", "quarter", "day_of_year", "is_weekend",
                   "trend_days", "year_sin", "year_cos"):
        frame[column] = calendar.map(lambda features: features[column])
    return frame.dropna(subset=FEATURES)


def _next_day_features(
    date: pd.Timestamp, store_id: int, values: list[float], start_date: pd.Timestamp
) -> dict[str, float]:
    if len(values) < max(LAGS):
        raise ValueError("Not enough history to create 28-day lag features.")
    features = _calendar_features(date, store_id, start_date)
    features.update({f"lag_{lag}": float(values[-lag]) for lag in LAGS})
    features.update({
        f"rolling_mean_{window}": float(np.mean(values[-window:]))
        for window in ROLLING_WINDOWS
    })
    return features


def recursive_forecast(
    model, history: pd.DataFrame, dates: pd.DatetimeIndex, stores: list[int], start_date: pd.Timestamp
) -> pd.DataFrame:
    """Forecast forward one day at a time, feeding predictions into later lag features."""
    values = {
        int(store_id): group.sort_values("date")["sales_revenue"].astype(float).tolist()
        for store_id, group in history.groupby("store_id")
    }
    rows = []
    for date in dates:
        feature_rows = [_next_day_features(date, store_id, values[store_id], start_date) for store_id in stores]
        predictions = np.maximum(model.predict(pd.DataFrame(feature_rows)[FEATURES]), 0.0)
        for store_id, prediction in zip(stores, predictions):
            prediction = float(prediction)
            rows.append({"date": date, "store_id": store_id, "forecast_revenue": prediction})
            values[store_id].append(prediction)
    return pd.DataFrame(rows)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.abs(actual).sum())
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "WAPE_pct": float(np.abs(actual - predicted).sum() / denominator * 100) if denominator else 0.0,
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }


def build_models() -> dict[str, object]:
    """Return the three regressors required by the project specification."""
    return {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Random Forest": RandomForestRegressor(
            n_estimators=160, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=220, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, objective="reg:squarederror",
            n_jobs=4, random_state=42, verbosity=0,
        ),
    }


def run_forecast(
    daily: pd.DataFrame, validation_days: int, forecast_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    if validation_days <= 0 or forecast_days <= 0:
        raise ValueError("Validation and forecast horizons must be positive whole numbers.")
    if len(dates) <= validation_days + max(LAGS):
        raise ValueError(
            f"Need more than {validation_days + max(LAGS)} daily observations; found {len(dates)}."
        )

    start_date = dates.min()
    cutoff = dates[-validation_days]
    features = make_training_features(daily, start_date)
    train = features.loc[features["date"] < cutoff]
    if train.empty:
        raise ValueError("No training rows remain before the validation period.")

    stores = sorted(int(value) for value in daily["store_id"].unique())
    validation_dates = pd.date_range(cutoff, dates.max(), freq="D")
    actual_validation = daily.loc[daily["date"].isin(validation_dates), ["date", "store_id", "sales_revenue"]]
    metrics = []
    for name, model in build_models().items():
        model.fit(train[FEATURES], train["sales_revenue"])
        predicted = recursive_forecast(
            model,
            daily.loc[daily["date"] < cutoff],
            validation_dates,
            stores,
            start_date,
        ).rename(columns={"forecast_revenue": "predicted_revenue"})
        compared = actual_validation.merge(predicted, on=["date", "store_id"], validate="one_to_one")
        metrics.append({"model": name, **_metrics(compared["sales_revenue"], compared["predicted_revenue"])})

    metrics_frame = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
    best_name = str(metrics_frame.iloc[0]["model"])
    best_model = build_models()[best_name]
    best_model.fit(features[FEATURES], features["sales_revenue"])
    future_dates = pd.date_range(dates.max() + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    forecast = recursive_forecast(best_model, daily, future_dates, stores, start_date)
    forecast["model"] = best_name
    return metrics_frame, forecast, features.assign(_best_model=best_name)


def build_report(
    script_file: str,
    transactions: pd.DataFrame,
    daily: pd.DataFrame,
    quality: dict[str, int],
    metrics: pd.DataFrame,
    forecast: pd.DataFrame,
    validation_days: int,
    data_dir: Path,
) -> None:
    report = GenAIReport("Restaurant Sales Forecasting")
    monthly = daily.set_index("date").groupby(pd.Grouper(freq="MS"))["sales_revenue"].sum().reset_index()
    monthly["period"] = monthly["date"].dt.strftime("%Y-%m")
    yearly = daily.assign(year=daily["date"].dt.year).groupby("year", as_index=False)["sales_revenue"].sum()
    quarterly = daily.assign(
        quarter=daily["date"].dt.to_period("Q").astype(str)
    ).groupby("quarter", as_index=False)["sales_revenue"].sum()
    store_sales = (
        transactions.groupby(["store_id", "restaurant_name"], as_index=False)["sales_revenue"].sum()
        .sort_values("sales_revenue", ascending=False)
    )
    item_sales = (
        transactions.groupby(["item_id", "name"], as_index=False)
        .agg(sales_revenue=("sales_revenue", "sum"), units_sold=("item_count", "sum"))
        .sort_values("sales_revenue", ascending=False)
        .head(15)
    )
    weekday = daily.assign(day_name=daily["date"].dt.day_name()).groupby(
        "day_name", as_index=False
    )["sales_revenue"].mean()
    weekday["day_name"] = pd.Categorical(
        weekday["day_name"],
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    weekday = weekday.sort_values("day_name")

    report.grid([
        report.kv("Data and forecast", {
            "source folder": str(data_dir),
            "transaction rows after cleaning": len(transactions),
            "restaurants": transactions["store_id"].nunique(),
            "items": transactions["item_id"].nunique(),
            "historical dates": f"{daily['date'].min().date()} to {daily['date'].max().date()}",
            "validation horizon (days)": validation_days,
            "future horizon (days)": forecast["date"].nunique(),
            "best model": metrics.iloc[0]["model"],
        }),
        report.table("Data quality checks", [{"check": key, "rows": value} for key, value in quality.items()]),
    ])
    report.full_table("Time-ordered model comparison (recursive validation)", metrics, rows=10)
    report.plots([
        (px.line(monthly, x="period", y="sales_revenue", title="Monthly restaurant revenue"),
         "Sales trend by month"),
        (px.bar(store_sales, x="restaurant_name", y="sales_revenue", title="Revenue by restaurant"),
         "Restaurant performance"),
        (px.bar(weekday, x="day_name", y="sales_revenue", title="Average revenue by weekday"),
         "Day-of-week pattern"),
        (px.bar(item_sales.sort_values("sales_revenue"), x="sales_revenue", y="name", orientation="h",
                title="Top 15 items by revenue"),
         "High-performing items"),
        (px.bar(quarterly, x="quarter", y="sales_revenue", title="Revenue by quarter"),
         "Quarterly trend"),
        (px.bar(yearly, x="year", y="sales_revenue", title="Revenue by year"),
         "Yearly trend"),
    ])
    report.full_table("Restaurant revenue totals", store_sales, rows=len(store_sales))
    report.full_table("Top items by revenue and units", item_sales, rows=15)
    monthly_forecast = forecast.assign(month=forecast["date"].dt.strftime("%Y-%m")).groupby(
        "month", as_index=False
    )["forecast_revenue"].sum()
    report.full_table("Next-horizon monthly forecast", monthly_forecast, rows=monthly_forecast.shape[0])
    report.full_text(
        "Data notes",
        "The provided item data has no category column, so category-level sales analysis cannot be "
        "computed without adding category labels. Revenue is calculated as transaction price × item count. "
        "Validation forecasts are recursive: each predicted day becomes lag history for the next day. "
        "This is a time-ordered backtest, not a random train/test split.",
    )
    report.save(script_file, "sales_forecasting_report.html")


def main() -> None:
    config = GenAIConfig.from_env()
    default_data_dir = config.dataset_dir / "Capstone 3"
    data_dir = Path(os.getenv("GENAI_SALES_DATA_DIR", str(default_data_dir))).expanduser()
    validation_days = int(os.getenv("GENAI_SALES_VALIDATION_DAYS", "28"))
    forecast_days = int(os.getenv("GENAI_SALES_HORIZON_DAYS", "365"))

    transactions, restaurants, quality = load_and_prepare(data_dir)
    daily = aggregate_daily_sales(transactions, restaurants)
    metrics, forecast, _ = run_forecast(daily, validation_days, forecast_days)

    artifact_dir = config.artifact_dir / "sales_forecasting"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(artifact_dir / "model_metrics.csv", index=False)
    forecast.to_csv(artifact_dir / "future_sales_forecast.csv", index=False)
    daily.to_csv(artifact_dir / "daily_sales.csv", index=False)

    complete_features = make_training_features(daily, daily["date"].min())
    best_name = str(metrics.iloc[0]["model"])
    best_model = build_models()[best_name]
    best_model.fit(complete_features[FEATURES], complete_features["sales_revenue"])
    joblib.dump(
        {"model": best_model, "model_name": best_name, "features": FEATURES,
         "forecast_unit": "daily revenue by restaurant"},
        artifact_dir / "best_sales_forecaster.joblib",
    )
    build_report(
        __file__, transactions, daily, quality, metrics, forecast, validation_days, data_dir
    )
    print(f"Best model: {best_name}")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
