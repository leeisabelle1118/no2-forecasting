from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.load_airnow import load_all
from daily_data import make_dataloaders


class DailyTransformer(nn.Module):
    """Compact transformer for univariate daily lagged forecasting."""

    def __init__(self, input_dim: int = 1, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        return self.head(h_last)


def evaluate(model: nn.Module, loader, device: str) -> tuple[float, float]:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    mae = nn.L1Loss(reduction="sum")
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_mse += float(mse(pred, yb).item())
            total_mae += float(mae(pred, yb).item())
            n += len(xb)
    return total_mse / max(1, n), total_mae / max(1, n)


def build_daily_series() -> pd.DataFrame:
    """Build the canonical daily NO2 mean series used for K=7, direct t+1 forecasting."""
    df_hourly = load_all()
    daily_mean = df_hourly.mean(axis=1).resample("D").mean().dropna()
    return pd.DataFrame({"date": daily_mean.index, "airnow_no2": daily_mean.values}).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=None,
        help="Optional CSV with columns: date, airnow_no2. If omitted, loads full AirNow daily series.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--train-end",
        type=str,
        default="auto",
        help="Last date included in training. Use 'auto' for first full-year chronological split.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    # Legacy per-model entrypoint delegates to canonical baseline runner.
    target = Path(__file__).resolve().parents[1] / "train_daily.py"
    cmd = [
        sys.executable,
        str(target),
        "--model",
        "transformer",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--results-dir",
        str(args.results_dir),
        "--train-end",
        str(args.train_end),
    ]
    if args.csv:
        cmd.extend(["--csv", args.csv])
    print("Delegating to baseline runner:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
