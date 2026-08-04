from __future__ import annotations

import argparse
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


def build_hourly_series() -> pd.DataFrame:
    """Build hourly mean NO2 across all AirNow sites from the full archive."""
    df_hourly = load_all()
    hourly_mean = df_hourly.mean(axis=1).dropna().sort_index()
    return pd.DataFrame({"date": hourly_mean.index, "airnow_no2": hourly_mean.values}).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=None,
        help="Optional hourly CSV with columns: date, airnow_no2. If omitted, loads full AirNow hourly series.",
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
    args = p.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        source = args.csv
    else:
        df = build_hourly_series()
        source = "AirNow NetCDF archive"

    dates = pd.to_datetime(df["date"])
    print(f"Loaded hourly NO2 from {source}: rows={len(df)}, range={dates.min()} to {dates.max()}")

    train_loader, test_loader, _ = make_dataloaders(
        df,
        batch_size=args.batch_size,
        train_end=args.train_end,
    )
    horizon_days = getattr(test_loader.dataset, "forecast_horizon_days", None)
    if horizon_days != 1:
        raise ValueError(f"Expected direct one-step (t+1) forecasting; got horizon_days={horizon_days}")

    split_train_end = getattr(train_loader.dataset, "split_train_end", None)
    test_target_dates = getattr(test_loader.dataset, "target_dates", None)
    if split_train_end is not None and test_target_dates is not None:
        test_target_dates = pd.Series(pd.to_datetime(test_target_dates))
        if not test_target_dates.empty and (test_target_dates <= pd.Timestamp(split_train_end)).any():
            bad = test_target_dates[test_target_dates <= pd.Timestamp(split_train_end)].iloc[0]
            raise ValueError(
                "Evaluation includes non-future target date. "
                f"Found target_date={pd.Timestamp(bad).date()} <= train_end={pd.Timestamp(split_train_end).date()}"
            )
    input_dim = int(train_loader.dataset.X.shape[-1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DailyTransformer(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)

        train_mse = running / max(1, n)
        test_mse, test_mae = evaluate(model, test_loader, device)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[Transformer] epoch {epoch:3d}/{args.epochs} "
                f"train_mse={train_mse:.4f} test_mse={test_mse:.4f} test_mae={test_mae:.4f}"
            )

    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "forecast_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "transformer_daily.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
