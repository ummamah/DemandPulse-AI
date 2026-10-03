"""DemandPulse AI - Data loading and preprocessing.

Loads the Kaggle "Retail Store Inventory and Demand Forecasting" dataset,
validates its schema, parses dates, removes duplicates and sorts
chronologically. No synthetic data is ever generated here.
"""

from __future__ import annotations

import pandas as pd

DATA_PATH = "data/retail_store_inventory.csv"

EXPECTED_COLUMNS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price",
    "Discount", "Weather Condition", "Promotion", "Competitor Pricing",
    "Seasonality", "Epidemic", "Demand",
]

# Columns that must never be used as forecasting predictors (leakage policy).
LEAKAGE_COLUMNS = ["Units Ordered", "Inventory Level", "Units Sold"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and validate the raw dataset, returning a clean DataFrame."""
    df = pd.read_csv(path)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    # Parse dates and sort chronologically within each store/product series.
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # Drop exact duplicate rows (there are none in this dataset, but stay safe).
    df = df.drop_duplicates().reset_index(drop=True)

    if df[["Demand"]].isna().any().any():
        raise ValueError("Target column 'Demand' contains missing values.")

    return df


def dataset_stats(df: pd.DataFrame) -> dict:
    """Return real statistics inspected from the downloaded data."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "n_days": int(df["Date"].nunique()),
        "n_stores": int(df["Store ID"].nunique()),
        "n_products": int(df["Product ID"].nunique()),
        "n_categories": int(df["Category"].nunique()),
        "n_regions": int(df["Region"].nunique()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "demand_min": float(df["Demand"].min()),
        "demand_max": float(df["Demand"].max()),
        "demand_mean": round(float(df["Demand"].mean()), 2),
    }


if __name__ == "__main__":
    data = load_data()
    stats = dataset_stats(data)
    for key, value in stats.items():
        print(f"{key}: {value}")