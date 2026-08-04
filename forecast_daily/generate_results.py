from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow imports from project root and local model folders.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.load_airnow import load_all
from gnn.train_gnn_daily import DailyGNN
from mamba.train_mamba_daily import DailyMambaLike
from transformer.daily_data import (
    LEAD_DAYS,
    make_dataloaders,
    prepare_series,
    resolve_train_end,
)
from transformer.train_transformer_daily import DailyTransformer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_daily_series() -> pd.DataFrame:
    """Create the canonical daily NO2 series used by forecast_daily models."""
    df_hourly = load_all()  # raw archive timestamps, columns: site codes
    daily_mean = df_hourly.mean(axis=1).resample("D").mean().dropna()
    daily = pd.DataFrame({"date": daily_mean.index, "airnow_no2": daily_mean.values})
    return daily.reset_index(drop=True)


def evaluate_scaled(model: nn.Module, loader, device: str) -> tuple[float, float]:
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


def predict_scaled(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().reshape(-1)
            true = yb.numpy().reshape(-1)
            preds.append(pred)
            trues.append(true)
    if not preds:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(preds), np.concatenate(trues)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_one_model(
    name: str,
    model: nn.Module,
    train_loader,
    test_loader,
    scaler,
    epochs: int,
    lr: float,
    device: str,
    delta_loss_weight: float,
) -> tuple[nn.Module, dict, pd.DataFrame]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history_train_mse: list[float] = []
    history_test_mse: list[float] = []
    history_test_mae: list[float] = []

    model = model.to(device)

    def temporal_delta_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Penalize day-to-day slope mismatch to reduce 1-day-late reactions.
        if pred.shape[0] < 2:
            return torch.zeros((), device=pred.device)
        pred_delta = pred[1:] - pred[:-1]
        target_delta = target[1:] - target[:-1]
        return F.l1_loss(pred_delta, target_delta)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_delta = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            point_loss = loss_fn(pred, yb)
            delta_loss = temporal_delta_loss(pred, yb)
            loss = point_loss + delta_loss_weight * delta_loss
            loss.backward()
            opt.step()
            running += float(point_loss.item()) * len(xb)
            running_delta += float(delta_loss.item()) * len(xb)
            n += len(xb)

        train_mse = running / max(1, n)
        train_delta_l1 = running_delta / max(1, n)
        test_mse, test_mae = evaluate_scaled(model, test_loader, device)
        history_train_mse.append(train_mse)
        history_test_mse.append(test_mse)
        history_test_mae.append(test_mae)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{name}] epoch {epoch:3d}/{epochs} "
                f"train_mse={train_mse:.4f} train_delta_l1={train_delta_l1:.4f} "
                f"test_mse={test_mse:.4f} test_mae={test_mae:.4f}"
            )

    pred_scaled, true_scaled = predict_scaled(model, test_loader, device)
    pred_ppb = scaler.inverse_transform(pred_scaled)
    true_ppb = scaler.inverse_transform(true_scaled)

    summary = {
        "model": name,
        "epochs": epochs,
        "test_mae_ppb": float(np.mean(np.abs(pred_ppb - true_ppb))),
        "test_rmse_ppb": rmse(pred_ppb, true_ppb),
        "train_mse_scaled_last": float(history_train_mse[-1]),
        "test_mse_scaled_last": float(history_test_mse[-1]),
        "test_mae_scaled_last": float(history_test_mae[-1]),
        "history": {
            "train_mse": history_train_mse,
            "test_mse": history_test_mse,
            "test_mae": history_test_mae,
        },
    }

    pred_df = pd.DataFrame(
        {
            "pred_ppb": pred_ppb,
            "actual_ppb": true_ppb,
        }
    )
    return model, summary, pred_df


def save_plots(
    results_dir: Path,
    merged_preds: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    actual_daily_df: pd.DataFrame,
    plot_shift_days: int = -1,
) -> None:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    actual_daily = actual_daily_df.copy()
    actual_daily["date"] = pd.to_datetime(actual_daily["date"])
    actual_daily = actual_daily.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    color_map = {
        "transformer": "tab:blue",
        "mamba": "tab:orange",
        "gnn": "tab:green",
    }

    aligned_preds: dict[str, pd.DataFrame] = {}
    for name, dfp in merged_preds.items():
        tmp = dfp.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        tmp["pred_ppb"] = pd.to_numeric(tmp["pred_ppb"], errors="coerce")

        unknown_target_dates = tmp.loc[~tmp["date"].isin(actual_daily["date"]), "date"]
        if not unknown_target_dates.empty:
            bad = pd.Timestamp(unknown_target_dates.iloc[0]).date()
            raise ValueError(
                f"Prediction target date outside daily timeline for {name}: {bad}"
            )

        aligned = actual_daily[["date"]].merge(
            tmp[["date", "pred_ppb"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        if len(aligned) != len(actual_daily) or not aligned["date"].equals(actual_daily["date"]):
            raise ValueError(
                f"Aligned prediction timeline mismatch for {name}: "
                f"aligned_rows={len(aligned)} actual_rows={len(actual_daily)}"
            )
        aligned_preds[name] = aligned

    # 1) Full daily timeline with forecast overlays.
    # `plot_shift_days` is visual-only; metrics remain computed on true target dates.
    plt.figure(figsize=(13, 5))
    plt.plot(actual_daily["date"], actual_daily["airnow_no2"], label="Actual", linewidth=2.0, color="black", alpha=0.85)
    for name, aligned in aligned_preds.items():
        plot_dates = aligned["date"] + pd.to_timedelta(plot_shift_days, unit="D")
        plt.plot(plot_dates, aligned["pred_ppb"], label=f"{name.title()} Pred", linewidth=1.6, alpha=0.9, color=color_map.get(name))
    plt.title("Forecast Daily: Full Timeline (Actual Daily vs Forecast Overlays)")
    plt.xlabel("Date")
    plt.ylabel("NO2 (ppb)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "timeseries_all_models.png", dpi=160)
    plt.close()

    # 1b) Daily line + prediction markers with optional visual shift.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        actual_daily["date"],
        actual_daily["airnow_no2"],
        label="Actual daily mean",
        color="dimgray",
        linewidth=1.8,
        alpha=0.85,
    )

    for name, aligned in aligned_preds.items():
        plot_dates = aligned["date"] + pd.to_timedelta(plot_shift_days, unit="D")
        ax.plot(
            plot_dates,
            aligned["pred_ppb"],
            label=f"{name.title()} daily forecast",
            linewidth=1.2,
            marker="o",
            markersize=3.2,
            alpha=0.9,
            color=color_map.get(name),
        )

    ax.set_title("Forecast Daily: Full Daily Series With Forecast Overlays")
    ax.set_xlabel("Date")
    ax.set_ylabel("NO2 (ppb)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()
    plt.savefig(plots_dir / "daily_timeseries_with_target_aligned_forecasts.png", dpi=160)
    plt.close(fig)

    # 1c) Zoomed 3-month (test target) view with optional visual shift.
    target_date_values: list[pd.Timestamp] = []
    for aligned in aligned_preds.values():
        valid_dates = aligned.loc[aligned["pred_ppb"].notna(), "date"]
        if not valid_dates.empty:
            target_date_values.extend(pd.to_datetime(valid_dates).tolist())

    if target_date_values:
        zoom_start = pd.Timestamp(min(target_date_values))
        zoom_end = pd.Timestamp(max(target_date_values))
        zoom_actual = actual_daily[(actual_daily["date"] >= zoom_start) & (actual_daily["date"] <= zoom_end)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            zoom_actual["date"],
            zoom_actual["airnow_no2"],
            label="Actual daily mean",
            color="black",
            linewidth=2.0,
            alpha=0.9,
        )

        for name, aligned in aligned_preds.items():
            zoom_pred = aligned[(aligned["date"] >= zoom_start) & (aligned["date"] <= zoom_end)]
            plot_dates = zoom_pred["date"] + pd.to_timedelta(plot_shift_days, unit="D")
            ax.plot(
                plot_dates,
                zoom_pred["pred_ppb"],
                label=f"{name.title()} daily forecast",
                linewidth=1.3,
                marker="o",
                markersize=3.0,
                alpha=0.9,
                color=color_map.get(name),
            )

        ax.set_title("Forecast Daily: 3-Month Test Window (Forecast Overlays)")
        ax.set_xlabel("Date")
        ax.set_ylabel("NO2 (ppb)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.tight_layout()
        plt.savefig(plots_dir / "timeseries_test_3_months.png", dpi=160)
        plt.close(fig)

    # 2) Per-model scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, (name, dfp) in zip(axes, merged_preds.items()):
        ax.scatter(dfp["actual_ppb"], dfp["pred_ppb"], s=16, alpha=0.65)
        low = min(dfp["actual_ppb"].min(), dfp["pred_ppb"].min())
        high = max(dfp["actual_ppb"].max(), dfp["pred_ppb"].max())
        ax.plot([low, high], [low, high], "r--", linewidth=1.1)
        ax.set_title(name.title())
        ax.set_xlabel("Actual (ppb)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted (ppb)")
    fig.suptitle("Prediction Scatter by Model")
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_all_models.png", dpi=160)
    plt.close(fig)

    # 3) Metrics bar chart
    x = np.arange(len(metrics_df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, metrics_df["test_mae_ppb"], width=width, label="MAE")
    ax.bar(x + width / 2, metrics_df["test_rmse_ppb"], width=width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["model"].tolist())
    ax.set_ylabel("Error (ppb)")
    ax.set_title("Test Error Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_bar.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Train daily models and generate forecast_daily result plots")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--delta-loss-weight",
        type=float,
        default=0.35,
        help="Weight for day-to-day slope matching loss to reduce lagged reactions.",
    )
    p.add_argument(
        "--train-end",
        type=str,
        default="auto",
        help="Last date included in training. Use 'auto' for first full-year chronological split.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot-shift-days",
        type=int,
        default=-1,
        help="Shift prediction dates in plots only (e.g., -1 for one-day left shift).",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Subdirectory under forecast_daily to write artifacts",
    )
    args = p.parse_args()

    set_seed(args.seed)

    root = Path(__file__).resolve().parent
    results_dir = root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Building canonical daily NO2 series for direct t+1 forecasting...")
    daily_df = build_daily_series()
    effective_train_end = resolve_train_end(prepare_series(daily_df), train_end=args.train_end)
    print(f"Using train_end={effective_train_end.date()} (chronological split)")
    daily_csv = results_dir / "airnow_no2_daily_mean.csv"
    daily_df.to_csv(daily_csv, index=False)
    print(f"Saved daily CSV: {daily_csv}")

    train_loader, test_loader, scaler = make_dataloaders(
        daily_df,
        batch_size=args.batch_size,
        train_end=args.train_end,
    )
    horizon_days = getattr(test_loader.dataset, "forecast_horizon_days", None)
    if horizon_days != 1:
        raise ValueError(f"Expected direct one-step (t+1) forecasting; got horizon_days={horizon_days}")

    input_dim = int(train_loader.dataset.X.shape[-1])
    target_dates = getattr(test_loader.dataset, "target_dates", None)
    if target_dates is None:
        raise ValueError("Missing target_dates metadata on test dataset for daily t+1 evaluation")
    target_dates = pd.Series(pd.to_datetime(target_dates)).reset_index(drop=True)

    if not target_dates.empty and (target_dates <= effective_train_end).any():
        bad = target_dates[target_dates <= effective_train_end].iloc[0]
        raise ValueError(
            "Evaluation includes non-future target date. "
            f"Found target_date={pd.Timestamp(bad).date()} <= train_end={effective_train_end.date()}"
        )
    if len(target_dates) != len(test_loader.dataset):
        raise ValueError(
            "One-step evaluation requires exactly one prediction target per test dataset row. "
            f"target_dates={len(target_dates)} dataset_rows={len(test_loader.dataset)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_map: dict[str, nn.Module] = {
        "transformer": DailyTransformer(input_dim=input_dim),
        "mamba": DailyMambaLike(input_dim=input_dim),
        "gnn": DailyGNN(input_dim=input_dim),
    }

    metrics_records: list[dict] = []
    merged_preds: dict[str, pd.DataFrame] = {}

    for name, model in model_map.items():
        print(f"\n=== Training {name} ===")
        trained_model, summary, pred_df = train_one_model(
            name=name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            scaler=scaler,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            delta_loss_weight=args.delta_loss_weight,
        )

        ckpt_dir = results_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{name}_daily.pt"
        torch.save(trained_model.state_dict(), ckpt_path)

        if len(target_dates) != len(pred_df):
            raise ValueError(
                f"Target-date length mismatch for {name}: dates={len(target_dates)} predictions={len(pred_df)}"
            )
        pred_df.insert(0, "date", target_dates.values)
        pred_path = results_dir / f"predictions_{name}.csv"
        pred_df.to_csv(pred_path, index=False)

        with open(results_dir / f"history_{name}.json", "w", encoding="utf-8") as f:
            json.dump(summary["history"], f, indent=2)

        summary_clean = {
            k: v for k, v in summary.items() if k != "history"
        }
        summary_clean["checkpoint"] = str(ckpt_path)
        summary_clean["predictions_csv"] = str(pred_path)
        metrics_records.append(summary_clean)

        merged_preds[name] = pred_df
        print(f"Saved checkpoint: {ckpt_path}")
        print(f"Saved predictions: {pred_path}")

    metrics_df = pd.DataFrame(metrics_records).sort_values("test_rmse_ppb").reset_index(drop=True)
    metrics_csv = results_dir / "metrics.csv"
    metrics_json = results_dir / "metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_df.to_json(metrics_json, orient="records", indent=2)

    save_plots(results_dir, merged_preds, metrics_df, daily_df, plot_shift_days=args.plot_shift_days)

    print("\nDone. Artifacts saved to:", results_dir)
    print("- Daily CSV")
    print("- Per-model checkpoints and predictions")
    print("- Histories")
    print("- Plots: timeseries, scatter, metrics bar")
    print("\nModel ranking by RMSE (ppb):")
    for i, row in metrics_df.iterrows():
        print(f"{i + 1}. {row['model']}: RMSE={row['test_rmse_ppb']:.3f}, MAE={row['test_mae_ppb']:.3f}")


if __name__ == "__main__":
    main()
