# Forecast Daily Results Guide

This folder stores outputs from daily univariate NO2 forecasting experiments
for three models:
- Transformer
- Mamba-like (GRU stand-in)
- GNN-style temporal model

## Training/Testing Methodology

All results in this folder were produced by running:

```bash
python forecast_daily/generate_results.py --epochs 50 --batch-size 32 --lr 1e-3 --train-ratio 0.8
```

Methodology pipeline:
1. Load hourly AirNow NO2 data from NetCDF files.
2. Aggregate to one daily series by taking the mean across sites and then daily averaging.
3. Build a univariate dataset with:
   - lookback window K = 7 days
   - forecast horizon H = 1 day
4. Split chronologically (no shuffle):
   - first 80% of days for training
   - last 20% of days for testing
5. Fit min-max scaling on training data only, then apply to train and test.
6. Train each model on the same split and evaluate on the same test period.
7. Save predictions, metrics, checkpoints, and plots.

## Model results (this run)

From metrics.csv:

| Model | Epochs | Test MAE (ppb) | Test RMSE (ppb) |
|---|---:|---:|---:|
| GNN | 50 | 0.745 | 0.930 |
| Mamba | 50 | 0.720 | 0.934 |
| Transformer | 50 | 0.919 | 1.118 |

Quick takeaways:
- GNN has the best RMSE (slightly fewer large misses).
- Mamba has the best MAE (best typical absolute error).
- Transformer is weaker than the other two in this specific run.

## Graphs and how to interpret them

### 1) Time-series comparison

![Time Series Comparison](plots/timeseries_all_models.png)

What this plot is:
- Black line: actual NO2 on the test period.
- Colored lines: each model prediction over the same dates.

How to interpret:
- Better models follow the black line shape, peaks, and dips.
- If peaks are shifted right, the model has lag.
- If peaks are too low/high, the model under/overestimates amplitude.

What to look for in this run:
- Mamba and GNN generally track observed changes more closely than Transformer.

### 2) Scatter (actual vs predicted)

![Scatter Comparison](plots/scatter_all_models.png)

What this plot is:
- One panel per model.
- X-axis: actual NO2.
- Y-axis: predicted NO2.
- Red dashed line: perfect prediction (y = x).

How to interpret:
- Points near the red line indicate accurate predictions.
- Wide spread means higher error variance.
- Systematic points below the line at high actual values indicate peak underprediction.

What to look for in this run:
- GNN and Mamba point clouds are tighter than Transformer overall.

### 3) Metrics bar chart

![Metrics Bar Chart](plots/metrics_bar.png)

What this plot is:
- Side-by-side MAE and RMSE bars for each model.

How to interpret:
- Lower MAE = better average day-to-day accuracy.
- Lower RMSE = fewer or smaller large errors.
- If RMSE is much larger than MAE, large outliers are likely present.

What to look for in this run:
- Mamba leads on MAE.
- GNN leads on RMSE.
- Transformer trails both error metrics.

## Key output files

- airnow_no2_daily_mean.csv: Daily series used for training/testing.
- predictions_transformer.csv, predictions_mamba.csv, predictions_gnn.csv: test predictions with dates and actuals.
- metrics.csv and metrics.json: summary metrics and ranking.
- history_*.json: per-epoch train/test curves.
- checkpoints/*.pt: trained model weights.
- plots/*.png: visual diagnostics.

## Re-running experiments

Default results folder:

```bash
python forecast_daily/generate_results.py --epochs 50
```

Longer run in a separate folder (recommended):

```bash
python forecast_daily/generate_results.py --epochs 200 --results-dir results_longrun
```
