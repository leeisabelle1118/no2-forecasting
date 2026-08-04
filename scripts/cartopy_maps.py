#!/usr/bin/env python
"""
cartopy_maps.py — Generate Cartopy maps for per-site NO₂ model evaluation.

Visualizes station-level results on a geographic map for the test period (2024-07-01 → 2024-09-30):
  • Per-site mean observed NO₂
  • Per-site mean predicted NO₂ (for each trained model)
  • Per-site MAE (Mean Absolute Error)
  • Per-site bias (predicted - observed)

Usage
-----
# After running compare.py:
python cartopy_maps.py

Outputs
-------
  outputs/cartopy_observed_no2.png          — Mean observed NO₂ concentration across test period
  outputs/cartopy_transformer_pred_no2.png  — Mean predicted NO₂ (Transformer)
  outputs/cartopy_mamba_pred_no2.png        — Mean predicted NO₂ (Mamba)
  outputs/cartopy_transformer_mae.png       — Per-site MAE (Transformer)
  outputs/cartopy_mamba_mae.png             — Per-site MAE (Mamba)
  outputs/cartopy_transformer_bias.png      — Per-site bias (Transformer)
  outputs/cartopy_mamba_bias.png            — Per-site bias (Mamba)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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


def denormalize(data: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    """Denormalize predictions and targets using training mean (PPB units)."""
    return data * train_mean[np.newaxis, np.newaxis, :]


def plot_observed_no2(targets: np.ndarray, train_mean: np.ndarray, sites: list[str], 
                      vmin_no2: float, vmax_no2: float, out: Path):
    """Map of mean observed NO₂ across the test period.
    
    Uses its own color scale (vmin_no2, vmax_no2) based on observed data range.
    """
    from data.load_airnow import site_meta, DATA_DIR

    meta = site_meta(DATA_DIR)
    common = [s for s in sites if s in meta.index]
    site_idx = [sites.index(s) for s in common]

    lats = meta.loc[common, "lat"].values
    lons = meta.loc[common, "lon"].values

    # Denormalize and compute per-site mean
    targets_ppb = denormalize(targets, train_mean)
    mean_obs = targets_ppb[:, 0, site_idx].mean(axis=0)  # (n_common,)

    # Create map
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                standard_parallels=(29.5, 45.5))
    fig, ax = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": proj},
    )

    # Map background
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#c8dff0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f1eb", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#c8dff0", zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#aaaaaa", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#666666", linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#444444", linewidth=0.7, zorder=5)

    # Site scatter (using shared color scale)
    sc = ax.scatter(
        lons, lats,
        c=mean_obs, cmap="viridis",
        s=24, alpha=0.85,
        edgecolors="k", linewidths=0.25,
        vmin=vmin_no2, vmax=vmax_no2,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.75, pad=0.02)
    cbar.set_label(f"Mean NO₂ (PPB) [{vmin_no2:.1f}–{vmax_no2:.1f}]", fontsize=9)
    ax.set_title("Observed NO₂ — Test period (2024-07-01 to 2024-09-30)", fontsize=11, pad=8)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.name}")


def plot_predicted_no2(model_name: str, preds: np.ndarray, train_mean: np.ndarray,
                       sites: list[str], vmin_no2: float, vmax_no2: float, out: Path):
    """Map of mean predicted NO₂ for a single model.
    
    Uses model-specific color scale (vmin_no2, vmax_no2) optimized for this model's predictions.
    """
    from data.load_airnow import site_meta, DATA_DIR

    meta = site_meta(DATA_DIR)
    common = [s for s in sites if s in meta.index]
    site_idx = [sites.index(s) for s in common]

    lats = meta.loc[common, "lat"].values
    lons = meta.loc[common, "lon"].values

    # Denormalize and compute per-site mean
    preds_ppb = denormalize(preds, train_mean)
    mean_pred = preds_ppb[:, 0, site_idx].mean(axis=0)

    # Create map
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                standard_parallels=(29.5, 45.5))
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": proj})

    # Map background
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#c8dff0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f1eb", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#c8dff0", zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#aaaaaa", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#666666", linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#444444", linewidth=0.7, zorder=5)

    # Site scatter (using shared color scale — same as observed for direct comparison)
    sc = ax.scatter(
        lons, lats,
        c=mean_pred, cmap="viridis",
        s=24, alpha=0.85,
        edgecolors="k", linewidths=0.25,
        vmin=vmin_no2, vmax=vmax_no2,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.75, pad=0.02)
    cbar.set_label(f"Mean NO₂ (PPB) [{vmin_no2:.1f}–{vmax_no2:.1f}]", fontsize=9)
    ax.set_title(f"{model_name.capitalize()} Predicted NO₂ — Test period", fontsize=11, pad=8)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.name}")


def plot_mae_normalized(model_name: str, preds: np.ndarray, targets: np.ndarray,
                        sites: list[str], vmax_mae: float, out: Path):
    """Map of per-site MAE in normalized units.
    
    Uses model-specific error scale (0 to vmax_mae) optimized for this model's errors.
    """
    from data.load_airnow import site_meta, DATA_DIR

    meta = site_meta(DATA_DIR)
    common = [s for s in sites if s in meta.index]
    site_idx = [sites.index(s) for s in common]

    lats = meta.loc[common, "lat"].values
    lons = meta.loc[common, "lon"].values

    # Compute MAE in normalized units (before denormalization)
    site_mae = np.abs(preds - targets).mean(axis=(0, 1))
    mae_common = site_mae[site_idx]

    # Create map
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                standard_parallels=(29.5, 45.5))
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": proj})

    # Map background
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#c8dff0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f1eb", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#c8dff0", zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#aaaaaa", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#666666", linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#444444", linewidth=0.7, zorder=5)

    # Site scatter (using shared error scale)
    sc = ax.scatter(
        lons, lats,
        c=mae_common, cmap="YlOrRd",
        s=24, alpha=0.85,
        edgecolors="k", linewidths=0.25,
        vmin=0, vmax=vmax_mae,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.75, pad=0.02)
    cbar.set_label(f"MAE (normalized) [0–{vmax_mae:.3f}]", fontsize=9)
    ax.set_title(f"{model_name.capitalize()} — Per-site MAE", fontsize=11, pad=8)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.name}")


def plot_bias(model_name: str, preds: np.ndarray, targets: np.ndarray,
              train_mean: np.ndarray, sites: list[str], vmax_bias: float, out: Path):
    """Map of per-site bias (predicted - observed) in PPB.
    
    Uses model-specific diverging scale centered at zero (±vmax_bias) optimized for this model's bias.
    """
    from data.load_airnow import site_meta, DATA_DIR

    meta = site_meta(DATA_DIR)
    common = [s for s in sites if s in meta.index]
    site_idx = [sites.index(s) for s in common]

    lats = meta.loc[common, "lat"].values
    lons = meta.loc[common, "lon"].values

    # Denormalize and compute bias
    preds_ppb = denormalize(preds, train_mean)
    targets_ppb = denormalize(targets, train_mean)
    bias = (preds_ppb - targets_ppb)[:, 0, site_idx].mean(axis=0)

    # Create map
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                standard_parallels=(29.5, 45.5))
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": proj})

    # Map background
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#c8dff0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f1eb", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#c8dff0", zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#aaaaaa", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#666666", linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#444444", linewidth=0.7, zorder=5)

    # Site scatter (diverging colormap centered at 0, symmetric scale)
    sc = ax.scatter(
        lons, lats,
        c=bias, cmap="RdBu_r",
        s=24, alpha=0.85,
        edgecolors="k", linewidths=0.25,
        vmin=-vmax_bias, vmax=vmax_bias,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.75, pad=0.02)
    cbar.set_label(f"Bias (PPB) [−{vmax_bias:.2f} to +{vmax_bias:.2f}]", fontsize=9)
    ax.set_title(f"{model_name.capitalize()} — Per-site Bias (Pred − Obs)", fontsize=11, pad=8)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.name}")


def main():
    from data.load_airnow import load_sequences, DATA_DIR, FULL_TRAIN_END, get_train_mean
    from models.transformer_no2 import NO2Transformer, evaluate
    from models.mamba_no2 import NO2Mamba

    OUTPUTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data
    print("Loading test data …")
    X, y, _, sites = load_sequences(seq_len=24, pred_len=6, norm_end=str(FULL_TRAIN_END))
    ts = np.arange(len(X))
    
    # Split to test (after FULL_TRAIN_END, which is at index ~8784)
    test_start = 8784  # approximate index after 12-month training window
    X_test = X[test_start:]
    y_test = y[test_start:]

    train_mean = get_train_mean()
    n_sites = len(sites)

    # ── Load all models and collect predictions ──────────────────────────
    print("Computing model-specific color scales …")
    
    models_config = [("transformer", NO2Transformer), ("mamba", NO2Mamba)]
    all_preds = {}
    
    for model_name, model_class in models_config:
        ckpt = _latest_ckpt(model_name)
        if ckpt is None:
            continue
        
        meta_hist = _load_history(ckpt)
        if not meta_hist:
            continue
        
        seq_len = meta_hist.get("seq_len", 24)
        pred_len = meta_hist.get("pred_len", 6)
        d_model = meta_hist.get("d_model", 128)
        n_layers = meta_hist.get("n_layers", 2)
        
        m = model_class(n_sites=n_sites, seq_len=seq_len,
                        pred_len=pred_len, d_model=d_model,
                        n_layers=n_layers)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        
        with torch.no_grad():
            preds, _ = evaluate(m, X_test, y_test, device=device)
        
        all_preds[model_name] = preds

    # ── Calculate observed color scales ────────────────────────────────────
    y_test_ppb = denormalize(y_test, train_mean)
    obs_values = y_test_ppb[:, 0, :].mean(axis=0)
    obs_vmin_no2 = 0.0
    obs_vmax_no2 = np.percentile(obs_values, 95)
    obs_vmax_mae = 0.0
    obs_vmax_bias = 0.0

    # ── Calculate model-specific color scales ─────────────────────────────
    model_scales = {}
    
    for model_name, preds in all_preds.items():
        # NO₂ scale for this model
        preds_ppb = denormalize(preds, train_mean)
        pred_values = preds_ppb[:, 0, :].mean(axis=0)
        vmin_no2 = 0.0
        vmax_no2 = np.percentile(pred_values, 95)
        
        # MAE scale for this model
        site_mae = np.abs(preds - y_test).mean(axis=(0, 1))
        vmax_mae = np.percentile(site_mae, 95)
        
        # Bias scale for this model (symmetric)
        bias = (preds_ppb - y_test_ppb)[:, 0, :].mean(axis=0)
        vmax_bias = np.percentile(np.abs(bias), 95)
        
        model_scales[model_name] = {
            "vmin_no2": vmin_no2,
            "vmax_no2": vmax_no2,
            "vmax_mae": vmax_mae,
            "vmax_bias": vmax_bias,
        }
        
        print(f"\n  {model_name.upper()}:")
        print(f"    NO₂ scale: {vmin_no2:.2f} – {vmax_no2:.2f} PPB")
        print(f"    MAE scale: 0 – {vmax_mae:.3f} (normalized)")
        print(f"    Bias scale: ±{vmax_bias:.2f} PPB")
    
    print(f"\n  OBSERVED:")
    print(f"    NO₂ scale: {obs_vmin_no2:.2f} – {obs_vmax_no2:.2f} PPB")

    # ── Generate visualizations with model-specific color scales ──────────
    print("\nGenerating Cartopy visualizations …")
    plot_observed_no2(y_test, train_mean, sites, obs_vmin_no2, obs_vmax_no2,
                     OUTPUTS / "cartopy_observed_no2.png")

    # Generate model-specific maps with their own color scales
    for model_name, preds in all_preds.items():
        scales = model_scales[model_name]
        print(f"\n  Creating maps for {model_name} …")
        plot_predicted_no2(model_name, preds, train_mean, sites, 
                          scales["vmin_no2"], scales["vmax_no2"],
                          OUTPUTS / f"cartopy_{model_name}_pred_no2.png")
        plot_mae_normalized(model_name, preds, y_test, sites, scales["vmax_mae"],
                           OUTPUTS / f"cartopy_{model_name}_mae.png")
        plot_bias(model_name, preds, y_test, train_mean, sites, scales["vmax_bias"],
                 OUTPUTS / f"cartopy_{model_name}_bias.png")

    print("\n✓ Cartopy map generation complete!")
    print(f"\n✓ Each model uses its own consistent color scales:")
    for model_name, scales in model_scales.items():
        print(f"\n  {model_name.upper()}:")
        print(f"    • NO₂ (viridis): {scales['vmin_no2']:.2f}–{scales['vmax_no2']:.2f} PPB")
        print(f"    • Errors (YlOrRd): 0–{scales['vmax_mae']:.3f} normalized")
        print(f"    • Bias (RdBu_r): ±{scales['vmax_bias']:.2f} PPB")


if __name__ == "__main__":
    main()
