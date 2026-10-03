"""DemandPulse AI - basic tests.

Run with:  python -m pytest tests/ -v   (or:  python tests/test_app.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd  # noqa: E402

from data_preprocessing import dataset_stats, load_data              # noqa: E402
from evaluate_model import evaluate, mae, rmse, wape                 # noqa: E402
from feature_engineering import (FEATURES, build_features,           # noqa: E402
                                 chronological_split_dates)
from forecasting import recursive_forecast                           # noqa: E402
from inventory_recommendation import (classify_stockout_risk,        # noqa: E402
                                      compute_recommendation)


# ---------------------------------------------------------------- dataset
def test_dataset_loads():
    df = load_data()
    assert len(df) == 76000
    assert not df.isna().any().any()
    assert df.duplicated().sum() == 0
    assert str(df["Date"].min().date()) == "2022-01-01"
    assert str(df["Date"].max().date()) == "2024-01-30"


def test_dataset_stats_real_values():
    stats = dataset_stats(load_data())
    assert stats["n_stores"] == 5
    assert stats["n_products"] == 20
    assert stats["missing_values"] == 0


# ------------------------------------------------------------ preprocessing
def test_features_built_and_leak_free():
    df = build_features(load_data())
    for col in ["lag_1", "lag_7", "lag_14", "lag_28",
                "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]:
        assert col in df.columns
        assert df[col].isna().sum() == 0  # rows needing unknown lags dropped
    # leakage columns must not be model features
    assert "Units Ordered" not in FEATURES
    assert "Inventory Level" not in FEATURES
    assert "Units Sold" not in FEATURES


def test_chronological_split_is_ordered():
    dates = pd.Series(pd.date_range("2022-01-01", periods=100))
    split = chronological_split_dates(dates)
    assert split["train_end"] < split["val_end"]


# -------------------------------------------------------------- forecasting
def test_recursive_forecast_shape_and_positive():
    df = load_data()
    combo = df[(df["Store ID"] == "S001") & (df["Product ID"] == "P0001")]
    bundle = None
    import joblib
    bundle = joblib.load(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "demand_forecasting_model.pkl"))
    fc = recursive_forecast(bundle["model"], combo, horizon=7)
    assert len(fc) == 7
    assert all(f["forecast"] >= 0 for f in fc)
    # dates must continue after the last historical date
    assert fc[0]["date"] > str(combo["Date"].max().date())


def test_forecast_invalid_selection_raises():
    df = load_data().head(10)  # too short history
    import joblib
    bundle = joblib.load(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "demand_forecasting_model.pkl"))
    try:
        recursive_forecast(bundle["model"], df, horizon=7)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


# ------------------------------------------------------------- inventory
def test_stockout_risk_classification():
    assert classify_stockout_risk(1000, 100, 10) == "LOW"
    assert classify_stockout_risk(80, 100, 10) == "MEDIUM"
    assert classify_stockout_risk(40, 100, 10) == "HIGH"


def test_recommendation_formula():
    forecast = [{"date": f"2024-02-{d:02d}", "forecast": 10.0} for d in range(1, 8)]
    rec = compute_recommendation(forecast, current_inventory=42,
                                 lead_time_days=3, recent_demand_std=5.0)
    assert rec["forecast_demand"] == 70.0
    assert rec["avg_daily_demand"] == 10.0
    assert rec["lead_time_demand"] == 30.0
    # safety stock = ceil(1.28 * 5 * sqrt(3)) = ceil(11.08...) = 12
    assert rec["safety_stock"] == 12
    # reorder = max(0, 70 + 30 + 12 - 42) = 70
    assert rec["recommended_reorder"] == 70
    assert rec["stockout_risk"] == "HIGH"


def test_recommendation_no_reorder_when_healthy():
    forecast = [{"date": "2024-02-01", "forecast": 5.0}] * 7
    rec = compute_recommendation(forecast, current_inventory=500,
                                 lead_time_days=3, recent_demand_std=1.0)
    assert rec["recommended_reorder"] == 0
    assert rec["stockout_risk"] == "LOW"


# -------------------------------------------------------------- metrics
def test_metrics_values():
    y, p = [10, 20, 30], [12, 18, 33]
    # errors: 2, 2, 3 -> MAE = 7/3, RMSE = sqrt(17/3), WAPE = 7/60
    assert abs(mae(y, p) - 7 / 3) < 1e-9
    assert abs(rmse(y, p) - (17 / 3) ** 0.5) < 1e-9
    assert abs(wape(y, p) - 7 / 60) < 1e-9
    result = evaluate(y, p)
    assert set(result) == {"MAE", "RMSE", "WAPE"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)