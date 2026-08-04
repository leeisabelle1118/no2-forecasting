# forecast_daily

Daily lagged NO2 forecasting pipeline using a strict direct one-step target:
- Lookback window K = 7 daily rows
- Forecast horizon H = 1 day (direct t+1)
- One scalar NO2 value per day as the only model input channel
- Chronological split with train end set by --train-end (default: auto first full year)
- Predictions aligned only to valid target dates in the test window

## Purpose

This folder is the daily baseline/benchmark track and is intentionally separate
from the hourly multi-site sequence models in the project root.

## Pipeline summary

1. Load AirNow archive values and aggregate to one canonical daily mean series.
2. Build lagged daily windows of length 7.
3. Predict the next daily value directly (no recursive multi-step rollout, no hourly-to-daily supervision).
4. Train Transformer, Mamba-like, and GNN-style daily models on the same split.
5. Save checkpoints, dated predictions, metrics, and aligned plots.

## Run training for one model

```bash
python forecast_daily/train_daily.py --model transformer --epochs 50 --train-end auto
python forecast_daily/train_daily.py --model mamba --epochs 50 --train-end auto
python forecast_daily/train_daily.py --model gnn --epochs 50 --train-end auto
```

## Run full results generation

```bash
python forecast_daily/generate_results.py --epochs 50 --batch-size 32 --lr 1e-3 --train-end auto
```

Outputs are written under forecast_daily/results by default.

## Plots

- plots/timeseries_all_models.png: full daily actual series with prediction overlays on valid target dates.
- plots/daily_timeseries_with_target_aligned_forecasts.png: full-year daily actual series with forecast markers only on valid target dates.
- plots/scatter_all_models.png: actual vs predicted scatter.
- plots/metrics_bar.png: MAE/RMSE comparison.

## Detailed run notes

See forecast_daily/results/README.md for per-run metric interpretation and artifact details.
