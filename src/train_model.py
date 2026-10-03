"""DemandPulse AI - Train, compare and save demand forecasting models.

Models compared
---------------
1. Seasonal Naive baseline  -> prediction = demand exactly 7 days earlier
2. Random Forest Regressor  -> pooled global model across all store/product
3. HistGradientBoostingRegressor -> pooled global model

Selection rule: the ML model with the lowest validation MAE is deployed;
metrics reported in the app/README come from the chronological TEST period.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import dataset_stats, load_data          # noqa: E402
from evaluate_model import evaluate                              # noqa: E402
from feature_engineering import (CATEGORICAL_FEATURES, FEATURES,  # noqa: E402
                                 NUMERIC_FEATURES, TARGET,
                                 build_features,
                                 chronological_split_dates)

MODELS_DIR = "models"


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_random_forest() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=5,
            n_jobs=-1, random_state=42)),
    ])


def make_hist_gb() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, max_depth=8,
            random_state=42)),
    ])


def main() -> None:
    print("=" * 60)
    print("DemandPulse AI - model training")
    print("=" * 60)

    raw = load_data()
    stats = dataset_stats(raw)
    print("Dataset stats:", json.dumps(stats, indent=2))

    # Chronological split: model is always evaluated on FUTURE observations.
    split = chronological_split_dates(raw["Date"], train_frac=0.75, val_frac=0.10)
    train_end, val_end = pd.Timestamp(split["train_end"]), pd.Timestamp(split["val_end"])
    print(f"Train: <= {train_end.date()} | Validation: <= {val_end.date()} "
          f"| Test: > {val_end.date()}")

    # Feature building uses only past data per row, so applying it to the
    # full panel is leak-free for one-step-ahead evaluation.
    featured = build_features(raw, dropna=True)

    train = featured[featured["Date"] <= train_end]
    val = featured[(featured["Date"] > train_end) & (featured["Date"] <= val_end)]
    test = featured[featured["Date"] > val_end]
    print(f"Rows  train={len(train)}  val={len(val)}  test={len(test)}")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = {}

    # ---- 1. Seasonal Naive baseline -------------------------------------
    results["Seasonal Naive (lag_7)"] = {
        "validation": evaluate(y_val, val["lag_7"]),
        "test": evaluate(y_test, test["lag_7"]),
    }

    # ---- 2/3. ML models --------------------------------------------------
    candidates = {"Random Forest": make_random_forest(),
                  "HistGradientBoosting": make_hist_gb()}

    best_name, best_val_mae, best_model = None, float("inf"), None
    for name, pipe in candidates.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        val_metrics = evaluate(y_val, pipe.predict(X_val))
        test_metrics = evaluate(y_test, pipe.predict(X_test))
        results[name] = {"validation": val_metrics, "test": test_metrics}
        if val_metrics["MAE"] < best_val_mae:
            best_name, best_val_mae, best_model = name, val_metrics["MAE"], pipe

    # ---- Model comparison table ------------------------------------------
    print("\nModel comparison (chronological test period)")
    print(f"{'Model':<26}{'MAE':>9}{'RMSE':>9}{'WAPE%':>9}")
    for name, r in results.items():
        t = r["test"]
        star = "  <-- selected" if name == best_name else ""
        print(f"{name:<26}{t['MAE']:>9.2f}{t['RMSE']:>9.2f}{t['WAPE']:>9.2f}{star}")

    # ---- Feature importance (only from the actual selected model) --------
    # RandomForest exposes feature_importances_ directly; for models without
    # it (e.g. HistGradientBoosting) we compute real permutation importance
    # on the validation period - no invented values.
    importances = {}
    model_step = best_model.named_steps.get("model")
    if hasattr(model_step, "feature_importances_"):
        prep = best_model.named_steps["prep"]
        names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
        pairs = sorted(zip(names, model_step.feature_importances_), key=lambda p: -p[1])[:12]
        importances = {n: round(float(v), 4) for n, v in pairs}
    else:
        from sklearn.inspection import permutation_importance

        print("Computing permutation importance on the validation period ...")
        perm = permutation_importance(best_model, X_val, y_val,
                                      n_repeats=5, random_state=42,
                                      scoring="neg_mean_absolute_error",
                                      n_jobs=-1)
        pairs = sorted(zip(FEATURES, perm.importances_mean), key=lambda p: -p[1])[:12]
        pairs = [(n, v) for n, v in pairs if v > 0]
        importances = {n: round(float(v), 3) for n, v in pairs}

    # ---- Persist artifacts -----------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "demand_forecasting_model.pkl")
    joblib.dump({"model": best_model,
                 "model_name": best_name,
                 "features": FEATURES}, model_path)

    # Per-combination statistics used at inference time.
    recent_std = (raw.sort_values("Date")
                  .groupby(["Store ID", "Product ID"])["Demand"]
                  .apply(lambda s: s.tail(28).std(ddof=0)))
    latest_inventory = (raw.sort_values("Date")
                        .groupby(["Store ID", "Product ID"])["Inventory Level"]
                        .last())
    latest_inputs = (raw.sort_values("Date")
                     .groupby(["Store ID", "Product ID"])
                     .tail(1)[["Store ID", "Product ID", "Category", "Region",
                               "Price", "Discount", "Promotion",
                               "Competitor Pricing"]])

    metadata = {
        "model_name": best_name,
        "results": results,
        "feature_importance": importances,
        "split": {"train_end": str(train_end.date()),
                  "val_end": str(val_end.date())},
        "dataset_stats": stats,
        "recent_demand_std": {f"{s}|{p}": round(float(v), 2)
                              for (s, p), v in recent_std.items()},
        "latest_inventory": {f"{s}|{p}": int(v)
                             for (s, p), v in latest_inventory.items()},
        "latest_inputs": latest_inputs.to_dict(orient="records"),
        "data_last_date": str(raw["Date"].max().date()),
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved metadata -> {meta_path}")
    print("Top features:", list(importances)[:5])


if __name__ == "__main__":
    main()