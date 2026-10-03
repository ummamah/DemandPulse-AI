"""DemandPulse AI - Recursive multi-step demand forecasting.

Produces the next ``horizon`` days of demand for one store/product
combination using ONLY historical information:

  * calendar features come from the future dates themselves,
  * lag / rolling features come from actual history plus previously
    predicted days (recursive strategy),
  * price / discount / promotion / competitor pricing are carried forward
    from the latest observed values (documented approximation),
  * no future sales, inventory, or Units Ordered information is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import FEATURES, add_date_features


def latest_business_inputs(history: pd.DataFrame) -> dict:
    """Latest observed planned inputs for a store/product combination."""
    last = history.sort_values("Date").iloc[-1]
    return {
        "Price": float(last["Price"]),
        "Discount": float(last["Discount"]),
        "Promotion": float(last["Promotion"]),
        "Competitor Pricing": float(last["Competitor Pricing"]),
        "Store ID": last["Store ID"],
        "Product ID": last["Product ID"],
        "Category": last["Category"],
        "Region": last["Region"],
    }


def recursive_forecast(model, history: pd.DataFrame, horizon: int = 7) -> list:
    """Forecast the next ``horizon`` days of demand for one combination.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    history : DataFrame with raw columns (Date, Demand, business inputs)
              for a single Store ID / Product ID, sorted by Date.
    horizon : number of future days to predict (default 7).

    Returns
    -------
    list of dicts: [{"date": ..., "forecast": ...}, ...]
    """
    if len(history) < 28:
        raise ValueError("Not enough historical data is available for this selection.")

    history = history.sort_values("Date").reset_index(drop=True)
    inputs = latest_business_inputs(history)
    known_demand = list(history["Demand"].astype(float))
    dates = list(pd.to_datetime(history["Date"]))

    forecasts = []
    for _ in range(horizon):
        next_date = dates[-1] + pd.Timedelta(days=1)
        dates.append(next_date)

        row = dict(inputs)
        row["Date"] = next_date

        # Lags come from actual history + previously predicted demand.
        for lag in (1, 7, 14, 28):
            row[f"lag_{lag}"] = known_demand[-lag]

        # Rolling means use the PREVIOUS days only (shift-by-one convention).
        for window in (7, 14, 28):
            row[f"rolling_mean_{window}"] = float(np.mean(known_demand[-window:]))

        row_df = add_date_features(pd.DataFrame([row]))
        X = row_df[FEATURES]
        prediction = float(max(0.0, model.predict(X)[0]))

        known_demand.append(prediction)
        forecasts.append({"date": next_date.strftime("%Y-%m-%d"),
                          "forecast": round(prediction, 1)})

    return forecasts


def recent_history(history: pd.DataFrame, days: int = 30) -> list:
    """Recent actual demand for charting, aligned with the forecast."""
    history = history.sort_values("Date").tail(days)
    return [{"date": d.strftime("%Y-%m-%d"), "demand": float(v)}
            for d, v in zip(history["Date"], history["Demand"])]