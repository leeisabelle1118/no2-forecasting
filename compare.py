#!/usr/bin/env python
"""
compare.py — Load saved checkpoints and compare Transformer vs Mamba.

Generates:
  • outputs/comparison_results.json  — MSE / MAE / param counts
  • outputs/comparison_curves.png    — Training loss curves
  • outputs/comparison_scatter.png   — Predicted vs actual NO₂ for each model
  • outputs/site_mae_map.png         — Per-site MAE on a lat/lon map

Usage
-----
# After training both models:
python train.py --model transformer
python train.py --model mamba
python compare.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"


def _latest_ckpt(model_name: str) -> Path | None:
    """Return the most recently modified checkpoint for a model."""
    pattern = str(OUTPUTS / f"{model_name}_*.pt")
    files = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    return Path(files[-1]) if files else None


def _load_history(ckpt_path: Path) -> dict:
    hist_path = Path(str(ckpt_path).replace(".pt", "_history.json"))
    if not hist_path.exists():
        return {}
    with open(hist_path) as f:
        return json.load(f)


def plot_curves(histories: dict[str, list[dict]], out: Path):
    """Plot training and validation MSE curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = {"transformer": "steelblue", "mamba": "darkorange"}
    for name, hist in histories.items():
        epochs    = [h["epoch"]    for h in hist]
        train_mse = [h["train_mse"] for h in hist]
        val_mse   = [h["val_mse"]   for h in hist]
        c = colors.get(name, "grey")
        axes[0].plot(epochs, train_mse, color=c, label=name.capitalize())
        axes[1].plot(epochs, val_mse,   color=c, label=name.capitalize())

    for ax, title in zip(axes, ["Training MSE", "Validation MSE"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (normalised)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Transformer vs Mamba — Training curves", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out.name}")


def plot_scatter(model_preds: dict[str, np.ndarray], targets: np.ndarray, out: Path):
    """Predicted vs actual scatter (flattened across time and sites)."""
    n_models = len(model_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    colors = {"transformer": "steelblue", "mamba": "darkorange"}
    flat_y = targets.ravel()
    # Sample for readability
    idx = np.random.choice(len(flat_y), min(5000, len(flat_y)), replace=False)
    flat_y = flat_y[idx]

    for ax, (name, preds) in zip(axes, model_preds.items()):
        flat_p = preds.ravel()[idx]
        c = colors.get(name, "grey")
        ax.scatter(flat_y, flat_p, alpha=0.15, s=6, color=c)
        lim = max(flat_y.max(), flat_p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="perfect")
        ax.set_xlabel("Actual NO₂ (normalised)")
        ax.set_ylabel("Predicted NO₂ (normalised)")
        ax.set_title(name.capitalize())
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend()

    fig.suptitle("Predicted vs Actual — test set", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out.name}")


def plot_site_mae(model_preds: dict[str, np.ndarray], targets: np.ndarray, out: Path):
    """Per-site MAE overlaid on a lat/lon scatter map."""
    from data.load_airnow import site_meta, DATA_DIR, load_sequences

    _, _, _, sites = load_sequences(seq_len=24, pred_len=6)   # for site codes
    meta = site_meta(DATA_DIR)
    common = [s for s in sites if s in meta.index]
    site_idx = [sites.index(s) for s in common]

    # MAE per site averaged over time and pred steps
    n_models = len(model_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5),
                              subplot_kw={"aspect": "auto"})
    if n_models == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, model_preds.items()):
        # preds: (n_windows, pred_len, n_sites)  targets same shape
        site_mae = np.abs(preds - targets).mean(axis=(0, 1))  # (n_sites,)
        mae_common = site_mae[site_idx]
        lats = meta.loc[common, "lat"].values
        lons = meta.loc[common, "lon"].values

        sc = ax.scatter(lons, lats, c=mae_common, cmap="YlOrRd",
                        s=40, alpha=0.85, edgecolors="k", linewidths=0.3,
                        vmin=0)
        plt.colorbar(sc, ax=ax, label="MAE (normalised)")
        ax.set_xlim(-130, -65); ax.set_ylim(24, 55)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"{name.capitalize()} — per-site MAE")
        ax.set_facecolor("#e8f4f8")

    fig.suptitle("Per-site test MAE — Transformer vs Mamba", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out.name}")


def main():
    import torch
    from data.load_airnow import load_sequences, DATA_DIR
    from models.transformer_no2 import NO2Transformer, evaluate
    from models.mamba_no2 import NO2Mamba

    OUTPUTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Find checkpoints ──────────────────────────────────────────────────────
    results, histories, model_preds = {}, {}, {}
    models_to_eval: list[tuple[str, nn.Module, Path]] = []

    for model_name in ("transformer", "mamba"):
        ckpt = _latest_ckpt(model_name)
        if ckpt is None:
            print(f"No checkpoint found for {model_name} — skipping.")
            continue
        meta = _load_history(ckpt)
        if not meta:
            print(f"No history JSON for {ckpt.name} — skipping.")
            continue

        seq_len  = meta.get("seq_len",  24)
        pred_len = meta.get("pred_len",  6)
        n_sites  = meta.get("n_sites", 197)
        d_model  = meta.get("d_model", 128)
        n_layers = meta.get("n_layers", 2)

        if model_name == "transformer":
            m = NO2Transformer(n_sites=n_sites, seq_len=seq_len,
                               pred_len=pred_len, d_model=d_model,
                               n_layers=n_layers)
        else:
            m = NO2Mamba(n_sites=n_sites, seq_len=seq_len,
                         pred_len=pred_len, d_model=d_model,
                         n_layers=n_layers)

        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        models_to_eval.append((model_name, m, meta))
        histories[model_name] = meta.get("history", [])
        results[model_name] = {
            "test_mse":  meta.get("test_mse"),
            "test_mae":  meta.get("test_mae"),
            "n_params":  meta.get("n_params"),
            "checkpoint": str(ckpt.name),
        }

    if not models_to_eval:
        print("No checkpoints found. Train at least one model first:\n"
              "  python train.py --model transformer\n"
              "  python train.py --model mamba")
        return

    # ── Load test data (same config as first checkpoint) ─────────────────────
    ref_meta = models_to_eval[0][2]
    print("Loading test data …")
    X, y, timestamps, sites = load_sequences(
        seq_len=ref_meta["seq_len"], pred_len=ref_meta["pred_len"])
    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    X_test  = X[n_train + n_val:]
    y_test  = y[n_train + n_val:]
    print(f"Test set: {len(X_test):,} windows\n")

    # ── Evaluate & collect predictions ────────────────────────────────────────
    import torch.nn as nn
    for model_name, m, meta in models_to_eval:
        m = m.to(device).eval()
        from models.transformer_no2 import _make_loader
        loader = _make_loader(X_test, y_test, 256, shuffle=False)
        preds_list = []
        with torch.no_grad():
            for xb, _ in loader:
                preds_list.append(m(xb.to(device)).cpu().numpy())
        model_preds[model_name] = np.concatenate(preds_list)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'Model':<15} {'Params':>10} {'Test MSE':>10} {'Test MAE':>10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name.capitalize():<15} {r['n_params']:>10,} "
              f"{r['test_mse']:>10.4f} {r['test_mae']:>10.4f}")

    with open(OUTPUTS / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → comparison_results.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if len(histories) > 1 or any(h for h in histories.values()):
        plot_curves(histories,   OUTPUTS / "comparison_curves.png")
    plot_scatter(model_preds, y_test, OUTPUTS / "comparison_scatter.png")
    try:
        plot_site_mae(model_preds, y_test, OUTPUTS / "site_mae_map.png")
    except Exception as e:
        print(f"(site MAE map skipped: {e})")


if __name__ == "__main__":
    main()
