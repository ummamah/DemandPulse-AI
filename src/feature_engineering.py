"""DemandPulse AI - Feature engineering.

Builds a leak-free supervised learning frame for 7-day demand forecasting.

Feature policy
--------------
ALLOWED (known at or before the forecast date):
  * calendar features derived from the date (weekday, month, ...)
  * historical lag features of Demand (lag_1, lag_7, lag_14, lag_28)
  * historical rolling means (rolling_mean_7/14/28, always shifted by 1 day
    so they only contain information strictly before the row's date)
  * price, discount, promotion, competitor pricing (planned business inputs)
  * static identifiers (store, product, category, region)

NOT ALLOWED (leakage):
  * same-day or future Units Sold / Demand
  * future Inventory Level
  * Units Ordered (a downstream inventory decision)
  * Weather Condition / Epidemic (unknown for future dates)
"""

from __future__ import annotations

import pandas as pd

TARGET = "Demand"

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

# Final predictor list used by the models.
NUMERIC_FEATURES = (
    ["Price", "Discount", "Promotion", "Competitor Pricing"]
    + [f"lag_{l}" for l in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + ["day", "weekday", "month", "quarter", "week_of_year", "is_weekend"]
)
CATEGORICAL_FEATURES = ["Store ID", "Product ID", "Category", "Region", "season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the Date column only (no leakage)."""
    df = df.copy()
    df["day"] = df["Date"].dt.day
    df["weekday"] = df["Date"].dt.weekday          # 0 = Monday
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["season"] = df["month"].map(
        {12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag / rolling features computed strictly from PAST demand.

    Works on the long panel: grouping by (Store ID, Product ID) guarantees
    that lags never cross product/store boundaries. Rolling windows use
    ``shift(1)`` so the value for day t is the mean of the *previous* w days.
    """
    df = df.copy()
    grouped = df.groupby(["Store ID", "Product ID"], sort=False)["Demand"]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
        )
    return df


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Full feature-building pipeline for training / evaluation frames."""
    df = add_date_features(df)
    df = add_lag_and_rolling_features(df)
    if dropna:
        # Rows without a full 28-day history cannot be used for supervised
        # learning; dropping them removes every row that would need unknown
        # lag information.
        needed = [f"lag_{l}" for l in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
        df = df.dropna(subset=needed).reset_index(drop=True)
    return df


def chronological_split_dates(dates: pd.Series,
                              train_frac: float = 0.75,
                              val_frac: float = 0.10) -> dict:
    """Return date boundaries for a chronological train/val/test split."""
    unique_dates = sorted(dates.unique())
    n = len(unique_dates)
    train_end = unique_dates[int(n * train_frac) - 1]
    val_end = unique_dates[int(n * (train_frac + val_frac)) - 1]
    return {"train_end": train_end, "val_end": val_end}


if __name__ == "__main__":
    from data_preprocessing import load_data

    data = load_data()
    featured = build_features(data)
    print("rows after feature building:", len(featured))
    print("feature columns:", FEATURES)
    split = chronological_split_dates(data["Date"])
    print("split boundaries:", split)