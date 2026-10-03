"""DemandPulse AI - Flask web application.

Single-page dashboard: select a store + product + forecast horizon, the app
runs the trained ML model and returns the forecast, inventory status,
recommendation, model metrics and feature importance.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from forecasting import recent_history, recursive_forecast
from inventory_recommendation import (DEFAULT_LEAD_TIME_DAYS,
                                      compute_recommendation)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "demand_forecasting_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "retail_store_inventory.csv")

VALID_HORIZONS = [7, 14]

app = Flask(__name__)

# ---------------------------------------------------------------- load once
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
MODEL_NAME = bundle["model_name"]

with open(META_PATH, encoding="utf-8") as fh:
    META = json.load(fh)

DATA = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values(
    ["Store ID", "Product ID", "Date"])

STORES = sorted(DATA["Store ID"].unique().tolist())
PRODUCTS = sorted(DATA["Product ID"].unique().tolist())
PRODUCT_CATEGORY = DATA.drop_duplicates("Product ID").set_index(
    "Product ID")["Category"].to_dict()


def _combo(store: str, product: str) -> pd.DataFrame:
    return DATA[(DATA["Store ID"] == store) & (DATA["Product ID"] == product)]


def _build_insight(rec: dict, history: pd.DataFrame) -> str:
    """Generate one concise, data-backed business insight."""
    recent = history.sort_values("Date")["Demand"].tail(14)
    last_week = recent.tail(7).mean()
    prev_week = recent.head(7).mean()
    trend = "trending upward" if last_week > prev_week * 1.05 else (
        "trending downward" if last_week < prev_week * 0.95 else "stable")

    gap = rec["expected_gap"]
    if rec["stockout_risk"] == "HIGH":
        action = ("Current inventory is below the expected requirement - "
                  "replenish before the forecast period begins.")
    elif rec["stockout_risk"] == "MEDIUM":
        action = "Inventory is close to expected demand - plan a reorder soon."
    else:
        action = "Inventory comfortably covers the forecast period."

    trend_part = f"Demand is {trend} " if trend != "stable" else "Demand is stable "
    trend_part += (f"(last-7-day average {last_week:.0f} vs previous-7-day "
                   f"average {prev_week:.0f} units). ")
    gap_part = (f"The expected inventory gap over the planning period is "
                f"{max(gap, 0):.0f} units. ") if gap > 0 else ""
    return (trend_part + gap_part + action).strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/options")
def options():
    return jsonify({
        "stores": STORES,
        "products": [{"id": p, "category": PRODUCT_CATEGORY[p]} for p in PRODUCTS],
        "horizons": VALID_HORIZONS,
        "default_horizon": 7,
    })


@app.route("/api/forecast")
def forecast_api():
    store = request.args.get("store", "").strip()
    product = request.args.get("product", "").strip()
    try:
        horizon = int(request.args.get("horizon", 7))
    except ValueError:
        return jsonify({"error": "Forecast horizon must be an integer."}), 400

    # ---- input validation (never crash on bad selections) ----------------
    if store not in STORES:
        return jsonify({"error": f"Unknown store '{store}'."}), 400
    if product not in PRODUCTS:
        return jsonify({"error": f"Unknown product '{product}'."}), 400
    if horizon not in VALID_HORIZONS:
        return jsonify({"error": f"Horizon must be one of {VALID_HORIZONS}."}), 400

    history = _combo(store, product)
    if len(history) < 28:
        return jsonify({"error": "Not enough historical data is available "
                                 "for this selection."}), 400

    try:
        forecast = recursive_forecast(MODEL, history, horizon=horizon)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    key = f"{store}|{product}"
    current_inventory = float(META["latest_inventory"].get(key, 0))
    recent_std = float(META["recent_demand_std"].get(key, 0.0))
    rec = compute_recommendation(forecast, current_inventory,
                                 lead_time_days=DEFAULT_LEAD_TIME_DAYS,
                                 recent_demand_std=recent_std)

    # Weekday seasonality pattern for this combination (actual averages).
    weekday_avg = (history.groupby(history["Date"].dt.weekday)["Demand"]
                   .mean().round(1))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_pattern = [{"day": day_names[i], "avg": float(weekday_avg.get(i, 0.0))}
                       for i in range(7)]

    return jsonify({
        "selection": {
            "store": store, "product": product,
            "category": PRODUCT_CATEGORY.get(product, ""),
        },
        "as_of_date": META["data_last_date"],
        "kpi": {
            "current_inventory": rec["current_inventory"],
            "forecast_demand": rec["forecast_demand"],
            "stockout_risk": rec["stockout_risk"],
            "risk_label": rec["risk_label"],
            "recommended_reorder": rec["recommended_reorder"],
            "horizon_days": rec["horizon_days"],
        },
        "history": recent_history(history, days=30),
        "forecast": forecast,
        "recommendation": rec,
        "inventory_status": {
            "current_stock": rec["current_inventory"],
            "expected_demand": rec["forecast_demand"],
            "safety_stock": rec["safety_stock"],
        },
        "model": {
            "selected": MODEL_NAME,
            "metrics": META["results"],
            "split": META["split"],
        },
        "feature_importance": META["feature_importance"],
        "weekday_pattern": weekday_pattern,
        "insight": _build_insight(rec, history),
    })


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)