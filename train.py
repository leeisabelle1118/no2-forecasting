#!/usr/bin/env python
"""
train.py — Train a Transformer or Mamba model on AirNow hourly NO₂ data.

Examples
--------
# Train Transformer (default) for 50 epochs, save to outputs/
python train.py

# Train Mamba with a 48-hour look-back and 12-hour forecast horizon
python train.py --model mamba --seq-len 48 --pred-len 12

# Resume / tune: lower LR, more epochs
python train.py --model transformer --lr 3e-4 --epochs 100 --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow running from the project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.load_airnow import load_sequences, DATA_DIR
from models.transformer_no2 import NO2Transformer, _make_loader, evaluate
from models.mamba_no2 import NO2Mamba
from models.gnn_no2 import NO2GNN, build_knn_adj

OUTPUTS = ROOT / "outputs"


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NO₂ Forecasting model")
    p.add_argument("--model",      choices=["transformer", "mamba", "gnn"],
                   default="transformer", help="Architecture to train")
    p.add_argument("--data-dir",   default=DATA_DIR, help="AirNow NetCDF folder")
    p.add_argument("--seq-len",    type=int, default=24,  help="Look-back window (hours)")
    p.add_argument("--pred-len",   type=int, default=6,   help="Forecast horizon (hours)")
    p.add_argument("--stride",     type=int, default=1,   help="Sliding window stride")
    p.add_argument("--d-model",    type=int, default=128, help="Model hidden dimension")
    p.add_argument("--n-layers",   type=int, default=None,
                   help="Encoder layers (default: 2 for Transformer/GNN, 3 for Mamba)")
    p.add_argument("--k-nn",       type=int, default=5,
                   help="k-nearest neighbours for GNN graph construction (default 5)")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int, default=8,
                   help="Early-stopping patience (epochs without val improvement)")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac",   type=float, default=0.15)
    p.add_argument("--fill-nan",   type=float, default=0.0)
    p.add_argument("--no-norm",    action="store_true",
                   help="Disable per-site normalisation")
    p.add_argument("--tag",        default="",
                   help="Extra string appended to the output filename")
    p.add_argument("--device",     default=None,
                   help="Force 'cpu' or 'cuda'. Auto-detected if omitted.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (extended from models/transformer_no2.py for checkpoint saving)
# ──────────────────────────────────────────────────────────────────────────────

def train_with_checkpointing(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: str,
    ckpt_path: Path,
) -> list[dict]:
    model = model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.MSELoss()
    loader  = _make_loader(X_train, y_train, batch_size, shuffle=True)

    history: list[dict] = []
    best_val, stale = float("inf"), 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        val_mse, val_mae = evaluate(model, X_val, y_val,
                                    batch_size=batch_size * 4, device=device)
        sched.step(val_mse)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_mse": train_loss,
               "val_mse": val_mse, "val_mae": val_mae,
               "lr": opt.param_groups[0]["lr"]}
        history.append(row)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_loss:.4f}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)")

        if val_mse < best_val:
            best_val = val_mse
            stale = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path.name}")
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stopping at epoch {epoch} (best val_mse={best_val:.4f})")
                break

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUTS.mkdir(exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print(f"  Seq/Pred: {args.seq_len}h look-back → {args.pred_len}h forecast")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data …")
    X, y, timestamps, sites = load_sequences(
        args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        fill_nan=args.fill_nan,
        normalize=not args.no_norm,
    )
    n_sites = X.shape[2]
    n       = len(X)
    n_train = int(n * args.train_frac)
    n_val   = int(n * args.val_frac)

    X_train, y_train = X[:n_train],              y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],        y[n_train+n_val:]

    print(f"  Train : {n_train:,} windows  "
          f"({timestamps[0]} → {timestamps[n_train-1]})")
    print(f"  Val   : {n_val:,} windows")
    print(f"  Test  : {len(X_test):,} windows\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    n_layers_default = {"transformer": 2, "mamba": 3, "gnn": 2}
    n_layers = args.n_layers or n_layers_default[args.model]

    if args.model == "transformer":
        model = NO2Transformer(n_sites=n_sites, seq_len=args.seq_len,
                               pred_len=args.pred_len, d_model=args.d_model,
                               n_layers=n_layers)
    elif args.model == "mamba":
        model = NO2Mamba(n_sites=n_sites, seq_len=args.seq_len,
                         pred_len=args.pred_len, d_model=args.d_model,
                         n_layers=n_layers)
    else:  # gnn
        from data.load_airnow import site_meta
        meta = site_meta(args.data_dir)
        adj  = build_knn_adj(meta["lat"].values, meta["lon"].values, k=args.k_nn)
        model = NO2GNN(n_sites=n_sites, seq_len=args.seq_len,
                       pred_len=args.pred_len, d_model=args.d_model,
                       n_layers=n_layers, k_nn=args.k_nn, adj=adj)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}\n")

    # ── Checkpoint name ───────────────────────────────────────────────────────
    tag = f"_{args.tag}" if args.tag else ""
    ckpt_name = (f"{args.model}_s{args.seq_len}_p{args.pred_len}"
                 f"_d{args.d_model}{tag}.pt")
    ckpt_path = OUTPUTS / ckpt_name

    # ── Train ─────────────────────────────────────────────────────────────────
    history = train_with_checkpointing(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, patience=args.patience,
        device=device, ckpt_path=ckpt_path,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_mse, test_mae = evaluate(model, X_test, y_test, device=device)
    print(f"\n{'='*60}")
    print(f"  TEST  mse={test_mse:.4f}   mae={test_mae:.4f}")
    print(f"{'='*60}\n")

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = OUTPUTS / ckpt_name.replace(".pt", "_history.json")
    meta = {
        "model": args.model, "n_sites": n_sites,
        "seq_len": args.seq_len, "pred_len": args.pred_len,
        "d_model": args.d_model, "n_layers": n_layers,
        "n_params": n_params,
        "k_nn": args.k_nn if args.model == "gnn" else None,
        "test_mse": test_mse, "test_mae": test_mae,
        "history": history,
    }
    with open(hist_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"History saved → {hist_path}")


if __name__ == "__main__":
    main()
