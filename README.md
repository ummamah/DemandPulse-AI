# DemandPulse AI

**Retail Demand Forecasting & Inventory Recommendations**

DemandPulse AI is a Machine Learning project that helps a retailer answer three
simple business questions:

1. **How much demand should we expect?** (7-day demand forecast)
2. **Are we likely to run out of stock?** (stockout risk)
3. **How much should we reorder?** (replenishment recommendation)

The application forecasts demand for a selected store/product combination and
converts the forecast into an easy-to-understand inventory recommendation,
shown in a single-page web dashboard.

## Overview

DemandPulse AI follows the pipeline:

```text
Data -> Preprocessing -> Feature Engineering -> Time-Based Split
     -> Baseline Forecast -> ML Forecast -> Evaluation
     -> Inventory Recommendation -> Web Dashboard
```

The ML model forecasts demand. A transparent, rule-based inventory engine
converts that forecast into a replenishment estimate. The two are clearly
separated: **model prediction** vs **business-rule recommendation**.

## Problem Statement

Retail businesses need to estimate future demand so they can avoid stockouts
and unnecessary excess inventory. Poor forecasts lead to missed sales when
stock runs out and higher holding costs when too much stock is purchased.

## Why Demand Forecasting Matters

- **Stockouts** result in missed sales and unhappy customers.
- **Excess inventory** increases holding costs and ties up working capital.
- **Better demand forecasts** support smarter, data-driven inventory decisions.

## Dataset

- **Name:** Retail Store Inventory and Demand Forecasting
- **Source:** https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting
- **License:** Apache 2.0 (per the Kaggle page). The dataset is **synthetically generated**.
- **File used:** `data/retail_store_inventory.csv`

Actual statistics inspected from the downloaded file:

| Statistic | Value |
|---|---|
| Rows | 76,000 |
| Columns | 16 |
| Date range | 2022-01-01 to 2024-01-30 (760 daily dates) |
| Stores | 5 (S001-S005) |
| Products | 20 (P0001-P0020) |
| Categories | 5 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Demand (target) | 4 to 430 units/day, mean ~104 |

Each store/product combination has exactly 760 daily observations.

## Features

Forecasting predictors (all available at or before the forecast date):

- **Calendar:** day, weekday, month, quarter, week-of-year, weekend flag, season
- **Lags:** lag_1, lag_7, lag_14, lag_28 (past demand only)
- **Rolling means:** rolling_mean_7/14/28 (always shifted by 1 day - past only)
- **Business inputs:** Price, Discount, Promotion, Competitor Pricing
- **Identifiers:** Store ID, Product ID, Category, Region

**Leakage policy - never used as predictors:** `Units Ordered` (a downstream
inventory decision), `Inventory Level`, `Units Sold`, and any future
information (including future weather/epidemic flags).

## Methodology

1. Load and validate the real downloaded dataset (schema, dates, duplicates, missing values).
2. Engineer calendar, lag and rolling features - strictly from historical data.
3. **Chronological split** (no random split, which would leak future information):
   - Train: 2022-01-01 to 2023-07-24 (75%)
   - Validation: to 2023-10-08 (10%)
   - Test: 2023-10-09 to 2024-01-30 (15%) - always *future* observations
4. Compare a **Seasonal Naive** baseline against **Random Forest** and
   **HistGradientBoosting** models; select the best by validation MAE.
5. Evaluate with **MAE**, **RMSE** and **WAPE** on the test period.
6. Convert the 7-day forecast into an inventory recommendation.

## Machine Learning Models

| Model | Description |
|---|---|
| Seasonal Naive (baseline) | Predicts demand equal to the value exactly 7 days earlier |
| Random Forest Regressor | Pooled global model across all store/product series |
| HistGradientBoostingRegressor | Gradient-boosted trees, pooled global model |

Evaluation is one-step-ahead over the chronological test window (each
prediction uses only past actuals); the dashboard's future forecast is
generated recursively for the next 7/14 days.

## Inventory Recommendation

The ML model forecasts demand; a rule-based engine turns it into a
recommendation (lead time = 3 days, configurable):

```text
Average Daily Forecast = Total Forecast Demand / Horizon
Lead Time Demand       = Average Daily Forecast x Lead Time
Safety Stock           = ceil(1.28 x Recent Demand Std x sqrt(Lead Time))   (~90% service level)
Expected Inventory Gap = Forecast Demand + Lead Time Demand + Safety Stock - Current Inventory
Recommended Reorder    = max(0, Expected Inventory Gap)
```

Stockout risk from the coverage ratio `inventory / (forecast demand + safety stock)`:

- `LOW` (>= 1.20): inventory comfortably exceeds expected demand -> Healthy Stock
- `MEDIUM` (0.70-1.20): inventory is close to expected demand -> Reorder Soon
- `HIGH` (< 0.70): expected demand exceeds current inventory -> High Stockout Risk

These are **decision-support estimates**, not guaranteed optimal order quantities.

## Results

Actual metrics from the chronological test period (2023-10-09 to 2024-01-30, 11,400 rows):

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| Seasonal Naive (lag_7) | 43.00 | 54.96 | 43.54% |
| Random Forest | 22.88 | 30.70 | 23.17% |
| **HistGradientBoosting (selected)** | **12.31** | **17.18** | **12.46%** |

HistGradientBoosting was selected because it achieved the lowest validation
MAE, and it clearly beats both the baseline and Random Forest on the test
period. `python src/train_model.py` reproduces the table above (fixed random_state).

## Project Structure

```text
DemandPulse-AI/
├── data/retail_store_inventory.csv      # real Kaggle dataset
├── models/demand_forecasting_model.pkl  # trained model + preprocessing
├── models/model_metadata.json           # metrics, importances, per-combo stats
├── notebooks/exploratory_analysis.ipynb # concise EDA
├── src/
│   ├── data_preprocessing.py            # load + validate + clean
│   ├── feature_engineering.py           # calendar/lag/rolling features (leak-free)
│   ├── train_model.py                   # train, compare, save models
│   ├── evaluate_model.py                # MAE / RMSE / WAPE
│   ├── forecasting.py                   # recursive 7/14-day forecast
│   └── inventory_recommendation.py      # stockout risk + reorder logic
├── templates/index.html                 # single-page dashboard
├── static/style.css, script.js, js/chart.umd.min.js
├── tests/test_app.py                    # basic tests
├── app.py                               # Flask application
├── requirements.txt
├── README.md
└── REPORT.md
```

## Installation

```bash
git clone <repository-url>
cd DemandPulse-AI

pip install -r requirements.txt
```

The dataset is already included at `data/retail_store_inventory.csv`. If you
need to re-download it manually from Kaggle, place the CSV at exactly that path.

## How to Run

```bash
# 1. (optional) retrain the models and regenerate metrics
python src/train_model.py

# 2. (optional) run the tests
python tests/test_app.py

# 3. start the web application
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Web Application

The single-page dashboard lets you:

1. Select a **Store**, **Product** (category shown automatically) and
   **Forecast Horizon** (7 or 14 days).
2. Click **Generate Forecast**.
3. See: current inventory, forecast demand, stockout risk, recommended
   reorder, the demand forecast chart (history + forecast), inventory status
   bars, the recommendation breakdown (lead-time demand, safety stock, gap),
   model performance (MAE/RMSE/WAPE for all models), feature importance and
   weekday seasonality.

Invalid selections return a clear message instead of crashing.

## Limitations

- The dataset is **synthetically generated**, so patterns differ from real stores.
- Recommendations are **decision-support estimates**, not guaranteed outcomes.
- Real deployment would require live inventory/sales data and actual lead times.
- Price/promotion for future days are carried forward from the latest known values.

## Future Improvements

- Real-time sales and inventory system integration
- Automated model retraining and monitoring
- Probabilistic forecasts (prediction intervals)
- Supplier lead-time optimisation
- Cloud deployment