"""DemandPulse AI - Evaluation metrics for demand forecasting."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error = sum|y-yhat| / sum|y|.

    Handles zero-demand days safely (no division by zero).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    total = np.sum(np.abs(y_true))
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total)


def mape(y_true, y_pred) -> float:
    """MAPE, only meaningful when no y_true == 0 (checked by caller)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MAPE undefined for zero-demand observations.")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(mae(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "WAPE": round(wape(y_true, y_pred) * 100, 2),
    }