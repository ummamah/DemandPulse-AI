# DemandPulse AI - Project Report

## 1. Project Title

**DemandPulse AI** - Retail Demand Forecasting & Inventory Recommendations

## 2. Introduction

Demand forecasting is the task of estimating how many units of a product
customers will want in the future. Retailers use these estimates to decide
when and how much stock to reorder. DemandPulse AI combines a machine
learning demand forecast with a simple, transparent inventory rule engine to
support those decisions.

## 3. Problem Statement

Retail businesses need to estimate future demand so they can avoid stockouts
and unnecessary excess inventory. Without a reliable forecast, a store either
runs out of popular products (missed sales) or holds too much stock (higher
holding costs). This project forecasts the next 7 days of demand for a
selected store/product combination and converts the forecast into a
replenishment recommendation.

## 4. Why the Project Was Made

- **Stockouts** can result in missed sales and dissatisfied customers.
- **Excess inventory** can increase holding costs and tie up capital.
- **Better demand forecasts** can support inventory decisions such as when
  and how much to reorder.

This project was created as an internship assignment to demonstrate an
end-to-end ML workflow: real data, feature engineering, time-series
validation, honest evaluation and a usable web interface.

## 5. Dataset

- **Name:** Retail Store Inventory and Demand Forecasting
- **URL:** https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting
- **License:** Apache 2.0 (per the Kaggle page). The dataset is **synthetically generated**.
- **File:** `data/retail_store_inventory.csv`

Statistics inspected from the actual downloaded file:

| Statistic | Value |
|---|---|
| Rows | 76,000 |
| Columns | 16 |
| Date range | 2022-01-01 to 2024-01-30 (760 daily dates) |
| Stores / Products | 5 stores (S001-S005), 20 products (P0001-P0020) |
| Categories / Regions | 5 categories, 4 regions |
| Missing values | 0 |
| Duplicate rows | 0 |
| Demand (target) | 4 to 430 units/day, mean ~104 |

Every store/product combination has exactly 760 daily rows. Major columns:
Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold,
Units Ordered, Price, Discount, Weather Condition, Promotion, Competitor
Pricing, Seasonality, Epidemic, Demand.

**Target variable:** `Demand` (next 7 days, per store/product).

## 6. Methodology

```text
Data
 ↓
Preprocessing        (parse dates, sort chronologically, check duplicates/missing)
 ↓
Feature Engineering  (calendar features, lags, rolling means - past only)
 ↓
Time-Based Split     (train 75% -> validation 10% -> test 15%, chronological)
 ↓
Baseline Forecast    (seasonal naive: demand 7 days earlier)
 ↓
ML Forecast          (Random Forest / HistGradientBoosting)
 ↓
Evaluation           (MAE, RMSE, WAPE on the future test period)
 ↓
Inventory Recommendation (lead-time demand, safety stock, reorder quantity)
 ↓
Web Dashboard        (single-page Flask + Chart.js application)
```

## 7. Forecasting Models

**Baseline - Seasonal Naive:** predicts that demand tomorrow equals demand
exactly 7 days earlier (lag_7). Simple, but ignores trends and drivers.

**Random Forest Regressor:** an ensemble of 200 decision trees trained on a
pooled global panel (all 100 store/product series together) with one-hot
encoded identifiers.

**HistGradientBoostingRegressor:** gradient-boosted trees, also pooled.
Pooled training lets the model learn shared patterns (price sensitivity,
weekday effects) across all series while identifiers distinguish them.

**Feature policy (no data leakage):**
- *Used:* date/calendar features, lag_1/7/14/28 of Demand, rolling_mean_7/14/28
  (shifted by one day), Price, Discount, Promotion, Competitor Pricing,
  Store/Product/Category/Region identifiers.
- *Never used:* `Units Ordered` (a downstream inventory decision), current or
  future `Units Sold`/`Demand`, future `Inventory Level`, and
  Weather/Epidemic (unknown for future dates).

## 8. Evaluation

Models are compared on a **chronological** validation period and reported on
a later **test** period (2023-10-09 to 2024-01-30, 11,400 rows). No random
splitting is used anywhere - a random split would let the model "see the
future" and report inflated accuracy.

Actual test results:

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| Seasonal Naive (lag_7) | 43.00 | 54.96 | 43.54% |
| Random Forest | 22.88 | 30.70 | 23.17% |
| **HistGradientBoosting (selected)** | **12.31** | **17.18** | **12.46%** |

MAPE is not reported as the headline metric, but zero-demand handling is
safe in WAPE (no division by zero). The selected model is HistGradientBoosting
because it achieved the lowest validation MAE and clearly the best test metrics.

## 9. Inventory Recommendation Logic

The ML model forecasts demand; a rule-based engine turns the forecast into a
replenishment estimate (lead time = 3 days, configurable):

```text
Average Daily Forecast = Total Forecast Demand / Horizon
Lead Time Demand       = Average Daily Forecast x Lead Time
Safety Stock           = ceil(1.28 x Recent Demand Std x sqrt(Lead Time))
Expected Inventory Gap = Forecast Demand + Lead Time Demand + Safety Stock - Current Inventory
Recommended Reorder    = max(0, Expected Inventory Gap)
```

- `1.28` is the z-score for a ~90% service level; `Recent Demand Std` is the
  standard deviation of the combination's last 28 days of actual demand.
- **Stockout risk** uses the coverage ratio
  `Current Inventory / (Forecast Demand + Safety Stock)`:
  - `LOW` if coverage >= 1.20 -> inventory comfortably exceeds expected demand
  - `MEDIUM` if 0.70 <= coverage < 1.20 -> inventory is close to expected demand
  - `HIGH` if coverage < 0.70 -> expected demand exceeds current inventory

These thresholds are documented business rules, not arbitrary claims, and the
recommendation is always presented as a **decision-support estimate**.

## 10. Web Application

The Flask application (`app.py`) serves one page (`templates/index.html`).
The user selects a Store, a Product and a forecast horizon (7 or 14 days) and
clicks **Generate Forecast**. The backend:

1. validates the selection (invalid inputs return a clear 400 message, never a crash);
2. builds the future feature frame using only historical information and
   recursively predicts the next days;
3. computes the inventory recommendation and stockout risk;
4. returns history (30 days), forecast, KPIs, model metrics, feature
   importance, weekday seasonality and a generated business insight.

The dashboard renders all of this in one desktop viewport.

## 11. Results

- HistGradientBoosting reduced test MAE to **12.31 units** (WAPE **12.46%**)
  versus **43.00 units** (WAPE 43.54%) for the seasonal-naive baseline -
  a ~71% reduction in average error.
- The forecast drives a working inventory workflow: for S001/P0001 the model
  forecasts ~716 units over 7 days against 383 units in stock, classifying
  HIGH stockout risk and recommending ~727 units of reorder (including
  lead-time demand and safety stock).
- Risk levels vary by selection (the dataset contains LOW, MEDIUM and HIGH
  cases), and all displayed values are computed live from the model.

## 12. Limitations

- The dataset is **synthetic**, so learned patterns do not reflect a real retailer.
- Recommendations are **decision-support estimates**, not guaranteed optimal orders;
  the system does not guarantee stockouts will be prevented.
- Real deployment would require live inventory/sales data feeds.
- Actual supplier lead times and constraints would need to be incorporated.
- Price/promotion for future days are carried forward from the latest known values.

## 13. Future Improvements

- Real-time sales and inventory system integration
- Automated retraining and model monitoring
- Probabilistic forecasting with prediction intervals
- Supplier lead-time optimisation
- Cloud deployment

## 14. Conclusion

DemandPulse AI demonstrates a complete, honest ML workflow on a real
downloaded Kaggle dataset: leak-free feature engineering, chronological
validation, a baseline-vs-ML comparison (MAE 43.00 -> 12.31), and a
transparent rule-based inventory engine, all presented in a single-page web
dashboard suitable for demonstration and further development.