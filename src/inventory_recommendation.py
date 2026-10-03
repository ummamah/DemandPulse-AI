"""DemandPulse AI - Inventory recommendation engine (decision-support).

This module converts an ML demand forecast into a replenishment
recommendation using transparent business rules. It is NOT an exact
procurement system - the output is a decision-support estimate.

Formulas
--------
    Average Daily Forecast  = Total Forecast Demand / Horizon
    Lead Time Demand        = Average Daily Forecast * Lead Time (days)
    Safety Stock            = ceil(1.28 * Recent Demand Std * sqrt(Lead Time))
                              (z = 1.28 corresponds to ~90% service level)
    Expected Inventory Gap  = Forecast Demand + Lead Time Demand
                              + Safety Stock - Current Inventory
    Recommended Reorder     = max(0, Expected Inventory Gap)

Stockout risk is classified from the coverage ratio:
    coverage = Current Inventory / (Forecast Demand + Safety Stock)
    coverage >= 1.20 -> LOW     (inventory comfortably covers demand)
    coverage >= 0.70 -> MEDIUM  (inventory is close to expected demand)
    coverage <  0.70 -> HIGH    (expected demand exceeds current inventory)
"""

from __future__ import annotations

import math
from typing import List

DEFAULT_LEAD_TIME_DAYS = 3
SERVICE_LEVEL_Z = 1.28  # ~90% service level

LOW_COVERAGE_THRESHOLD = 1.20
MEDIUM_COVERAGE_THRESHOLD = 0.70

RISK_LABELS = {
    "LOW": "🟢 Healthy Stock",
    "MEDIUM": "🟡 Reorder Soon",
    "HIGH": "🔴 High Stockout Risk",
}


def classify_stockout_risk(current_inventory: float, total_demand: float,
                           safety_stock: float) -> str:
    """Classify stockout risk from the documented coverage-ratio thresholds."""
    required = max(total_demand + safety_stock, 1e-9)
    coverage = current_inventory / required
    if coverage >= LOW_COVERAGE_THRESHOLD:
        return "LOW"
    if coverage >= MEDIUM_COVERAGE_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def compute_recommendation(forecast: List[dict],
                           current_inventory: float,
                           lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
                           recent_demand_std: float = 0.0) -> dict:
    """Turn a daily demand forecast into an inventory recommendation."""
    daily = [float(f["forecast"]) for f in forecast]
    horizon = len(daily)
    if horizon == 0:
        raise ValueError("Forecast horizon must contain at least one day.")

    total_demand = sum(daily)
    avg_daily = total_demand / horizon
    lead_time_demand = avg_daily * lead_time_days
    safety_stock = math.ceil(SERVICE_LEVEL_Z * max(recent_demand_std, 0.0)
                             * math.sqrt(lead_time_days))
    expected_gap = total_demand + lead_time_demand + safety_stock - current_inventory
    reorder_qty = max(0, math.ceil(expected_gap))
    risk = classify_stockout_risk(current_inventory, total_demand, safety_stock)

    return {
        "current_inventory": round(float(current_inventory)),
        "horizon_days": horizon,
        "forecast_demand": round(total_demand, 1),
        "avg_daily_demand": round(avg_daily, 1),
        "lead_time_days": lead_time_days,
        "lead_time_demand": round(lead_time_demand, 1),
        "safety_stock": safety_stock,
        "expected_gap": round(expected_gap, 1),
        "recommended_reorder": reorder_qty,
        "stockout_risk": risk,
        "risk_label": RISK_LABELS[risk],
        "reason": _reason_text(risk, current_inventory, total_demand),
    }


def _reason_text(risk: str, current_inventory: float, total_demand: float) -> str:
    if risk == "HIGH":
        return ("Forecasted demand is expected to exceed the available inventory "
                "within the planning period. Replenishment is recommended before "
                "the forecasted demand period.")
    if risk == "MEDIUM":
        return ("Current inventory is close to the expected demand. A reorder is "
                "advised soon to stay above the safety-stock buffer.")
    return ("Current inventory comfortably exceeds the expected demand for the "
            "planning period. No immediate replenishment is required.")