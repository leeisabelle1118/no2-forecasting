# Forecast Daily Results Guide

This folder stores model outputs for daily univariate NO2 forecasting.

## Files in this folder

- `airnow_no2_daily_mean.csv`: Daily NO2 series used for training/testing (derived from hourly AirNow site means).
- `predictions_transformer.csv`, `predictions_mamba.csv`, `predictions_gnn.csv`: Test-period predictions and actual values.
- `metrics.csv`, `metrics.json`: Summary error metrics for each model.
- `history_transformer.json`, `history_mamba.json`, `history_gnn.json`: Per-epoch learning curves.
- `checkpoints/*.pt`: Saved model weights.
- `plots/*.png`: Visual diagnostics and model comparison graphs.

## How to interpret each graph

### 1) `plots/timeseries_all_models.png`

What it shows:
- Black line = observed NO2 values on the test period.
- Colored lines = each model forecast at the same dates.

How to read it:
- Better models track the shape and turning points of the black line.
- Look for lag: if a model consistently peaks after the observed peak, it is reacting late.
- Look for amplitude bias: if peaks are always too small or too large, the model is under/over-sensitive.

What good performance looks like:
- Predicted curves overlap the observed curve for both high and low NO2 periods.
- Fewer large misses during spikes.

### 2) `plots/scatter_all_models.png`

What it shows:
- Each panel is one model.
- X-axis = actual NO2, Y-axis = predicted NO2.
- Red dashed diagonal = perfect predictions (`y = x`).

How to read it:
- Points close to the diagonal mean accurate predictions.
- Vertical spread around the line means noise/error.
- Pattern below line at high actual values means underprediction of peaks.
- Pattern above line at low actual values means overprediction of lows.

What good performance looks like:
- Tight point cloud around the dashed line across the full value range.
- Minimal systematic tilt/bias.

### 3) `plots/metrics_bar.png`

What it shows:
- MAE and RMSE bars for each model.

Metric meaning:
- MAE: average absolute error in ppb (lower is better).
- RMSE: penalizes larger errors more strongly (lower is better).

How to read it:
- Use MAE for typical day-to-day error.
- Use RMSE to judge robustness against large misses/spikes.
- If RMSE is much higher than MAE, the model likely has occasional big errors.

## Interpreting model performance overall

Use all three views together:
1. Start with `metrics_bar.png` to identify best aggregate error.
2. Confirm behavior in `timeseries_all_models.png` to check temporal tracking and lag.
3. Use `scatter_all_models.png` to identify bias (under/overprediction) and outliers.

A model can have similar MAE but very different behavior on spikes. Prefer the model that keeps both low aggregate error and better spike tracking for your use case.

## Re-running and generating new result sets

Baseline run (writes to `forecast_daily/results`):

```bash
python forecast_daily/generate_results.py --epochs 50
```

Longer run to a separate folder (recommended to avoid overwriting):

```bash
python forecast_daily/generate_results.py --epochs 200 --results-dir results_longrun
```
