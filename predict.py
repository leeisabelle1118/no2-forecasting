#!/usr/bin/env python
"""
predict.py — Load a trained checkpoint and forecast NO₂ for a date range.

Loads the most recent checkpoint for a model, runs it on the chosen date
range from the AirNow dataset, and saves:
  • outputs/predictions_<model>_<start>_<end>.csv  — hourly predictions per site
  • outputs/forecast_<site>.png                    — time-series plot for each
                                                     requested site

Examples
--------
# Forecast the last week of the test set using the best Transformer checkpoint
python predict.py --model transformer

# Forecast a specific date range and plot three sites by name
python predict.py --model mamba --start 2024-08-01 --end 2024-08-07 \\
    --sites "Seattle-Beacon Hill" "Portland SE Lafayette" "Sacramento Del Paso Manor"

# List available site names
python predict.py --list-sites
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
OUTPUTS = ROOT / "outputs"


def _latest_ckpt(model_name: str) -> Path | None:
    files = sorted(glob.glob(str(OUTPUTS / f"{model_name}_*.pt")),
                   key=lambda p: Path(p).stat().st_mtime)
    return Path(files[-1]) if files else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NO₂ forecast from a saved checkpoint")
    p.add_argument("--model",  choices=["transformer", "mamba"],
                   default="transformer")
    p.add_argument("--data-dir", default="/mnt/data3/AirNow")
    p.add_argument("--start",  default=None,
                   help="Forecast start date YYYY-MM-DD (default: start of test set)")
    p.add_argument("--end",    default=None,
                   help="Forecast end date YYYY-MM-DD (default: end of test set)")
    p.add_argument("--sites",  nargs="*", default=None,
                   help="Site names to plot (partial match). Plots all if omitted.")
    p.add_argument("--max-plots", type=int, default=6,
                   help="Maximum number of site plots to save (default 6)")
    p.add_argument("--list-sites", action="store_true",
                   help="Print all available site names and exit")
    return p.parse_args()


def main():
    import torch
    from data.load_airnow import load_all, load_sequences, site_meta, DATA_DIR
    from models.transformer_no2 import NO2Transformer, _make_loader
    from models.mamba_no2 import NO2Mamba

    args = parse_args()
    data_dir = args.data_dir or DATA_DIR

    # ── List sites and exit ───────────────────────────────────────────────────
    if args.list_sites:
        meta = site_meta(data_dir)
        for code, row in meta.iterrows():
            print(f"  {code}  {row['name']:40s}  {row['agency']}")
        return

    OUTPUTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Find checkpoint ───────────────────────────────────────────────────────
    ckpt = _latest_ckpt(args.model)
    if ckpt is None:
        print(f"No checkpoint found for '{args.model}'. Train first:\n"
              f"  python train.py --model {args.model}")
        return

    import json
    hist_path = Path(str(ckpt).replace(".pt", "_history.json"))
    meta_info = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    seq_len  = meta_info.get("seq_len",  24)
    pred_len = meta_info.get("pred_len",  6)
    d_model  = meta_info.get("d_model", 128)
    n_layers = meta_info.get("n_layers",  2)

    print(f"Checkpoint : {ckpt.name}")
    print(f"Config     : seq_len={seq_len}  pred_len={pred_len}  device={device}\n")

    # ── Load full dataset (raw, for plotting) and normalised sequences ────────
    df_raw = load_all(data_dir)                          # (n_hours, n_sites) PPB
    X, y, timestamps, sites = load_sequences(
        data_dir, seq_len=seq_len, pred_len=pred_len, normalize=True)

    n = len(X)
    n_train = int(n * 0.70); n_val = int(n * 0.15)
    n_sites = X.shape[2]

    # Default: test set
    if args.start:
        start_dt = pd.Timestamp(args.start)
        idx0 = next((i for i, t in enumerate(timestamps) if t >= start_dt), n_train + n_val)
    else:
        idx0 = n_train + n_val

    if args.end:
        end_dt = pd.Timestamp(args.end) + pd.Timedelta(hours=23)
        idx1 = next((i for i, t in enumerate(timestamps) if t > end_dt), n)
    else:
        idx1 = n

    X_pred = X[idx0:idx1]
    y_true = y[idx0:idx1]
    ts_pred = timestamps[idx0:idx1]
    print(f"Predicting {len(X_pred):,} windows  "
          f"({ts_pred[0]}  →  {ts_pred[-1]})\n")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.model == "transformer":
        model = NO2Transformer(n_sites=n_sites, seq_len=seq_len,
                               pred_len=pred_len, d_model=d_model,
                               n_layers=n_layers)
    else:
        model = NO2Mamba(n_sites=n_sites, seq_len=seq_len,
                         pred_len=pred_len, d_model=d_model,
                         n_layers=n_layers)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device).eval()

    # ── Run inference ─────────────────────────────────────────────────────────
    loader = _make_loader(X_pred, y_true, batch_size=256, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for xb, _ in loader:
            preds_list.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds_list)           # (n_windows, pred_len, n_sites)

    # ── Save predictions CSV ──────────────────────────────────────────────────
    # Flatten: one row per (window_start, horizon_step, site)
    rows = []
    for i, t0 in enumerate(ts_pred):
        for h in range(pred_len):
            target_time = t0 + pd.Timedelta(hours=seq_len + h)
            for s, code in enumerate(sites):
                rows.append({
                    "window_start": t0,
                    "target_time":  target_time,
                    "horizon_h":    h + 1,
                    "site":         code,
                    "predicted_norm": float(preds[i, h, s]),
                    "actual_norm":    float(y_true[i, h, s]),
                })
    df_out = pd.DataFrame(rows)
    start_tag = str(ts_pred[0].date()).replace("-", "")
    end_tag   = str(ts_pred[-1].date()).replace("-", "")
    csv_path  = OUTPUTS / f"predictions_{args.model}_{start_tag}_{end_tag}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved predictions → {csv_path.name}  ({len(df_out):,} rows)")

    # ── Site plots ────────────────────────────────────────────────────────────
    meta = site_meta(data_dir)

    if args.sites:
        # Partial-name match
        plot_codes = [
            c for c in sites
            if c in meta.index and any(
                q.lower() in meta.loc[c, "name"].lower() for q in args.sites
            )
        ]
        if not plot_codes:
            print("No sites matched the given names. Use --list-sites to see options.")
            plot_codes = sites[:args.max_plots]
    else:
        # Default: sites with best coverage
        coverage = df_raw[sites].notna().mean()
        plot_codes = coverage.nlargest(args.max_plots).index.tolist()

    print(f"\nGenerating plots for {len(plot_codes)} site(s) …")

    # Recover per-site training mean for de-normalisation
    arr_all = load_all(data_dir).fillna(0).clip(lower=0).values.astype("float32")
    split   = int(len(arr_all) * 0.8)
    train_mean = arr_all[:split].mean(axis=0)                      # (n_sites,)
    train_mean = np.where(train_mean == 0, 1.0, train_mean)

    for code in plot_codes:
        if code not in sites:
            continue
        s_idx  = sites.index(code)
        name   = meta.loc[code, "name"] if code in meta.index else code
        t_mean = train_mean[s_idx]

        # Predicted and actual, de-normalised to PPB
        pred_ppb = preds[:, 0, s_idx] * t_mean      # first horizon step
        true_ppb = y_true[:, 0, s_idx] * t_mean
        times    = [t + pd.Timedelta(hours=seq_len) for t in ts_pred]

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(times, true_ppb,  color="steelblue", lw=0.8, label="Actual",    alpha=0.85)
        ax.plot(times, pred_ppb,  color="tomato",    lw=0.8, label="Predicted", alpha=0.85)
        ax.set_title(f"{name} ({code}) — {args.model.capitalize()} 1-hour-ahead forecast")
        ax.set_ylabel("NO₂ (PPB)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (len(times)//300))))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()

        plot_path = OUTPUTS / f"forecast_{code}_{args.model}.png"
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        print(f"  Saved → {plot_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not installed. Run with the cartopy conda env:\n"
              "  mamba run -n cartopy python predict.py")
        raise SystemExit(1)
    main()
