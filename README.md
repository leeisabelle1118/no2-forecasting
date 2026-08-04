# NO2 Forecasting Baseline

This repository is now centered on one canonical baseline pipeline for daily NO2 forecasting.

## Baseline contract

The baseline is defined by forecast_daily/daily_data_contract.py and enforced end-to-end:

- One daily series with columns: date, airnow_no2
- Chronological split only (no random split)
- Default split boundary: auto = first full year for training, remainder for test
- Lookback window K = 7 days
- Forecast horizon H = 1 day (direct t+1 only)
- Scaling fit on training segment only
- Strict shape checks for model inputs and outputs
- Strict date alignment checks for evaluation

## Canonical entrypoint

Use forecast_daily/generate_results.py for all baseline runs.

Examples:

```bash
# Run all three baseline models (transformer, mamba, gnn)
python forecast_daily/generate_results.py --train-end auto

# Run one model only
python forecast_daily/generate_results.py --models transformer --train-end auto

# Use a custom CSV dataset
python forecast_daily/generate_results.py --csv path/to/daily_no2.csv --models gnn
```

Optional CSV must contain:

- date: parseable datetime
- airnow_no2: numeric daily NO2 value

## Standardized outputs

By default, outputs are written to forecast_daily/results.

Expected files:

- metrics.csv
- metrics.json
- predictions_transformer.csv
- predictions_mamba.csv
- predictions_gnn.csv
- history_transformer.json
- history_mamba.json
- history_gnn.json
- checkpoints/transformer_daily.pt
- checkpoints/mamba_daily.pt
- checkpoints/gnn_daily.pt
- plots/timeseries_all_models.png
- plots/daily_timeseries_with_target_aligned_forecasts.png
- plots/timeseries_test_3_months.png
- plots/scatter_all_models.png
- plots/metrics_bar.png

## Guardrails

The baseline runner now checks:

- Dataset schema integrity
- Duplicate or non-monotonic dates
- Strict train-before-test chronology
- Direct one-step horizon consistency
- Input/output tensor shapes
- Prediction/target shape compatibility
- Target date uniqueness and future-only evaluation

If any contract is violated, execution stops with an explicit error.

## Legacy paths (quarantined)

The old hourly/multi-site experimentation path is no longer the canonical workflow.

Files train.py, predict.py, and compare.py remain as legacy wrappers that delegate to the daily baseline runner. They are kept only for compatibility and should not be treated as independent pipelines.

Additional legacy artifacts are quarantined under forecast_daily/legacy_experiments.

## Repository layout (baseline-relevant)

```text
NO2 Forecasting/
├── forecast_daily/
│   ├── daily_data_contract.py
│   ├── generate_results.py
│   ├── train_daily.py
│   ├── transformer/
│   ├── mamba/
│   ├── gnn/
│   ├── results/
│   └── legacy_experiments/
├── data/
│   └── load_airnow.py
├── train.py      # legacy wrapper
├── predict.py    # legacy wrapper
└── compare.py    # legacy wrapper
```

## Reproducibility

Use the same environment and seed for consistent baseline comparisons.

```bash
python forecast_daily/generate_results.py --seed 42 --train-end auto
```

## Notes

- The baseline is intentionally simple and strict.
- Any future experimental path should live outside the canonical baseline flow and should not modify the data contract.
