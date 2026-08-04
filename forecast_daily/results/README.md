# Forecast Daily Results Guide

This folder stores outputs from daily univariate NO2 forecasting experiments
for three models:
- Transformer
- Mamba-like (GRU stand-in)
- GNN-style temporal model

## Training/Testing Methodology

All results in this folder were produced by running:

```bash
python forecast_daily/generate_results.py --epochs 50 --batch-size 32 --lr 1e-3 --train-end auto
```

Methodology pipeline:
1. Load hourly AirNow NO2 data from NetCDF files and compute the hourly mean across sites.
2. Build direct one-step daily supervision samples:
   - **Predictor (X):** one full day of 24 hourly NO2 values for day d — shape `(24, 1)`
   - **Target (y):** daily mean NO2 for day d+1
   - **Mapping:** day 0 hourly values → predict day 1, day 1 hourly values → predict day 2, …
   - One prediction target per valid day; no recursive multi-step rollout.
3. Split chronologically by target date using the first full calendar year (`--train-end auto`):
   - For the current AirNow range, training resolves to target dates 2023-07-02 through 2024-06-30.
   - Testing is the remaining tail (target dates 2024-07-01 onward).
   - The first input day (2023-07-01) is a predictor day only; its target is 2023-07-02.
4. Fit min-max scaling on training data only (inputs and targets together), then apply to train and test.
5. Train each model on the same split and evaluate on the same test period.
6. Save predictions keyed to their actual target dates, metrics, checkpoints, and plots.

## Guardrails

The pipeline enforces the following assertions at runtime and will raise `ValueError` if violated:
- `forecast_horizon_days == 1`: exactly one-step-ahead (t+1) direct forecasting.
- `len(target_dates) == len(dataset)`: one prediction per test sample.
- All test target dates are strictly after `train_end`.
- Train and test splits are both non-empty and time-ordered.
- Train targets and test targets do not overlap.

## Plots

### 1) Full-year time-series comparison

![Time Series Comparison](plots/timeseries_all_models.png)

What this plot shows:
- **Black line:** actual daily mean NO2 across the full available timeline (train + test).
- **Colored lines:** each model's one-step-ahead predictions, plotted only on their target dates.
  Days with no prediction (i.e., before the first test target date) appear as gaps — this is correct.

How to interpret:
- Better models follow the black line's shape, peaks, and dips over the test window.
- If forecast peaks are shifted right relative to actual peaks, the model has temporal lag.
- If peaks are too low or high, the model is under/over-estimating amplitude.

### 1b) Daily forecast overlay with markers

![Hourly Time Series](plots/hourly_timeseries_with_daily_forecasts.png)

What this plot shows:
- **Gray line:** actual daily mean NO2 across the full timeline.
- **Colored lines with markers:** each model's daily one-step forecast on target dates only.

How to interpret:
- Markers should begin at the first valid test target date and align exactly with the actual series date axis.
- No right-shift should be present: each predicted point sits on the date it was forecast for.

### 2) Scatter (actual vs predicted)

![Scatter Comparison](plots/scatter_all_models.png)

What this plot shows:
- One panel per model.
- X-axis: actual NO2 (ppb).
- Y-axis: predicted NO2 (ppb).
- Red dashed line: perfect prediction (y = x).

How to interpret:
- Points near the red line indicate accurate predictions.
- Wide spread means higher error variance.
- Points systematically below the line at high actual values indicate peak underprediction.

### 3) Metrics bar chart

![Metrics Bar Chart](plots/metrics_bar.png)

What this plot shows:
- Side-by-side MAE and RMSE bars for each model.

How to interpret:
- Lower MAE = better average day-to-day accuracy.
- Lower RMSE = fewer or smaller large errors.
- If RMSE is much larger than MAE, large outliers are likely present.

## Key output files

- `airnow_no2_daily_mean.csv`: Daily mean NO2 series (used for the actual line in plots).
- `predictions_transformer.csv`, `predictions_mamba.csv`, `predictions_gnn.csv`: test predictions with target dates and actuals.
- `metrics.csv` and `metrics.json`: summary metrics and model ranking.
- `history_*.json`: per-epoch train/test loss curves.
- `checkpoints/*.pt`: trained model weights.
- `plots/*.png`: visual diagnostics.

## Re-running experiments

Default results folder:

```bash
python forecast_daily/generate_results.py --epochs 50
```

Longer run in a separate folder (recommended for experiments):

```bash
python forecast_daily/generate_results.py --epochs 200 --results-dir results_longrun
```

Note: experiment subdirectories (`results_*/`) are excluded from git by `.gitignore`.

