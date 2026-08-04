"""PyTorch Dataset/DataLoader for direct one-step daily supervision from hourly NO2.

Setup implemented:
- Input DataFrame has columns: 'date', 'airnow_no2' at hourly resolution.
- One sample per input day d using 24 hourly values: X shape (24, 1).
- One-step daily target: y is daily mean for day d+1.
- Chronological train/test split by target day (no random shuffle).
- Min-max scaling fit on TRAIN ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


HOURS_PER_DAY = 24
INPUT_DAYS = 1
LOOKBACK_DAYS = INPUT_DAYS  # Backward-compatible import name.
LEAD_DAYS = 1
FULL_YEAR_END = pd.Timestamp("2024-06-30")


@dataclass
class MinMaxScaler1D:
    """Simple 1D min-max scaler fit only on training values."""

    min_: float
    max_: float

    @classmethod
    def fit(cls, x_train: np.ndarray) -> "MinMaxScaler1D":
        return cls(min_=float(np.min(x_train)), max_=float(np.max(x_train)))

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        if denom == 0.0:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - self.min_) / denom).astype(np.float32)

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        return (x_scaled * (self.max_ - self.min_) + self.min_).astype(np.float32)


class AirNowNO2Dataset(Dataset):
    """Daily-sample dataset: day d hourly sequence -> day d+1 daily target."""

    def __init__(self, X_scaled: np.ndarray, y_scaled: np.ndarray, target_dates: pd.DatetimeIndex):
        if X_scaled.ndim != 3:
            raise ValueError("X_scaled must have shape (N, 24, 1)")
        if X_scaled.shape[1] != HOURS_PER_DAY:
            raise ValueError(f"Expected {HOURS_PER_DAY} hourly steps per sample, got {X_scaled.shape[1]}")
        if X_scaled.shape[2] != 1:
            raise ValueError(f"Expected one NO2 channel per hourly step, got {X_scaled.shape[2]}")
        if y_scaled.ndim != 1:
            raise ValueError("y_scaled must be 1D before conversion to (N, 1)")
        if len(X_scaled) != len(y_scaled) or len(y_scaled) != len(target_dates):
            raise ValueError("X, y, and target_dates must have matching lengths")

        self.X = torch.from_numpy(X_scaled.astype(np.float32)).float()
        self.y = torch.from_numpy(y_scaled.astype(np.float32)[:, None]).float()
        self.target_dates = pd.to_datetime(target_dates)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def prepare_series(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and enforce chronological order for hourly NO2 series."""
    required = {"date", "airnow_no2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").dropna(subset=["airnow_no2"]).reset_index(drop=True)
    return out


def _build_daily_hourly_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Return complete-day hourly NO2 matrix with shape (n_days, 24)."""
    hourly = df.set_index("date")["airnow_no2"].astype(np.float32).resample("h").mean().dropna()

    temp = hourly.to_frame(name="airnow_no2")
    temp["day"] = temp.index.floor("D")
    temp["hour"] = temp.index.hour

    pivot = temp.pivot(index="day", columns="hour", values="airnow_no2")
    pivot = pivot.reindex(columns=list(range(HOURS_PER_DAY)))
    pivot = pivot.dropna(axis=0, how="any")

    if len(pivot) < INPUT_DAYS + LEAD_DAYS + 1:
        raise ValueError(
            "Not enough complete daily hourly blocks for direct one-step setup. "
            f"Need at least {INPUT_DAYS + LEAD_DAYS + 1} complete days; found {len(pivot)}."
        )

    return pivot.to_numpy(dtype=np.float32), pd.DatetimeIndex(pivot.index)


def resolve_train_end(df: pd.DataFrame, train_end: str | pd.Timestamp | None = "auto") -> pd.Timestamp:
    """Resolve train-end boundary on calendar days."""
    if train_end is None or (isinstance(train_end, str) and train_end.lower() == "auto"):
        min_day = pd.Timestamp(df["date"].min()).floor("D")
        max_day = pd.Timestamp(df["date"].max()).floor("D")

        inferred = min_day + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        if inferred >= max_day:
            raise ValueError(
                "Unable to infer a full-year train/test split from hourly data. "
                "Need at least 1 year plus 1 later day for test. "
                f"Found range {min_day.date()} to {max_day.date()}."
            )
        return inferred

    return pd.Timestamp(train_end)


def chronological_split(df: pd.DataFrame, train_end: str | pd.Timestamp | None = "auto") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by target date: train <= train_end, test > train_end."""
    train_end_ts = pd.Timestamp(train_end)
    train_df = df[df["target_date"] <= train_end_ts].copy()
    test_df = df[df["target_date"] > train_end_ts].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Chronological split produced an empty train or test set. "
            f"train_end={train_end_ts}, samples={len(df)}"
        )
    return train_df, test_df


def make_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    train_end: str | pd.Timestamp | None = "auto",
    lookback: int = LOOKBACK_DAYS,
    lead: int = LEAD_DAYS,
    include_time_features: bool = False,
    include_weather_features: bool = False,
    weather_feature_cols: List[str] | None = None,
) -> Tuple[DataLoader, DataLoader, MinMaxScaler1D]:
    """Build direct one-step dataloaders: day d hourly values -> day d+1 target."""
    if include_time_features or include_weather_features or weather_feature_cols:
        raise ValueError("This direct hourly setup uses hourly NO2 only; disable extra feature flags")
    if lookback != INPUT_DAYS:
        raise ValueError(f"Direct daily supervision expects lookback={INPUT_DAYS}, got {lookback}")
    if lead != LEAD_DAYS:
        raise ValueError(f"forecast_daily supports direct one-step forecasting only (lead={LEAD_DAYS}); got lead={lead}")

    df = prepare_series(df)
    daily_hourly, day_index = _build_daily_hourly_matrix(df)

    # Supervision: day d hourly -> day d+1 daily mean target.
    X_raw = daily_hourly[:-LEAD_DAYS][:, :, None]  # (N, 24, 1)
    y_raw = daily_hourly[LEAD_DAYS:].mean(axis=1)  # (N,)
    target_dates = day_index[LEAD_DAYS:]

    samples = pd.DataFrame({"target_date": pd.to_datetime(target_dates), "row": np.arange(len(target_dates))})
    effective_train_end = resolve_train_end(df, train_end=train_end)
    train_df, test_df = chronological_split(samples, train_end=effective_train_end)

    if not train_df["target_date"].is_monotonic_increasing or not test_df["target_date"].is_monotonic_increasing:
        raise ValueError("Train/test splits must be time-ordered by target date")
    if pd.Timestamp(train_df["target_date"].max()) >= pd.Timestamp(test_df["target_date"].min()):
        raise ValueError(
            "Train/test split is not strictly chronological: "
            f"train_max={pd.Timestamp(train_df['target_date'].max())}, "
            f"test_min={pd.Timestamp(test_df['target_date'].min())}"
        )

    tr_idx = train_df["row"].to_numpy(dtype=int)
    te_idx = test_df["row"].to_numpy(dtype=int)

    X_train_raw = X_raw[tr_idx]
    X_test_raw = X_raw[te_idx]
    y_train_raw = y_raw[tr_idx]
    y_test_raw = y_raw[te_idx]

    train_reference = np.concatenate([X_train_raw.reshape(-1), y_train_raw], axis=0)
    scaler = MinMaxScaler1D.fit(train_reference)

    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    y_train = scaler.transform(y_train_raw)
    y_test = scaler.transform(y_test_raw)

    train_ds = AirNowNO2Dataset(X_train, y_train, pd.DatetimeIndex(train_df["target_date"]))
    test_ds = AirNowNO2Dataset(X_test, y_test, pd.DatetimeIndex(test_df["target_date"]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_ds.feature_names = [f"hour_{h:02d}" for h in range(HOURS_PER_DAY)]
    test_ds.feature_names = [f"hour_{h:02d}" for h in range(HOURS_PER_DAY)]
    train_ds.split_train_end = pd.Timestamp(train_df["target_date"].max())
    test_ds.split_test_start = pd.Timestamp(test_df["target_date"].min())
    train_ds.forecast_horizon_days = LEAD_DAYS
    test_ds.forecast_horizon_days = LEAD_DAYS
    train_ds.forecast_mode = "direct_one_day_hourly_to_next_day_target"
    test_ds.forecast_mode = "direct_one_day_hourly_to_next_day_target"

    if train_ds.y.shape[1] != 1 or test_ds.y.shape[1] != 1:
        raise ValueError("Direct one-step setup must produce scalar daily targets with shape (N, 1)")

    return train_loader, test_loader, scaler


def _demo() -> None:
    """Small runnable demo with synthetic hourly data."""
    ts = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    no2 = 20 + 5 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24) + np.random.normal(0, 0.6, len(ts))
    df = pd.DataFrame({"date": ts, "airnow_no2": no2})

    train_loader, test_loader, scaler = make_dataloaders(df, batch_size=16, train_end="2024-01-30")
    xb, yb = next(iter(train_loader))
    print(f"Train batch X shape: {tuple(xb.shape)}")  # (batch, 24, 1)
    print(f"Train batch y shape: {tuple(yb.shape)}")  # (batch, 1)
    print(f"First target date: {train_loader.dataset.target_dates[0].date()}")
    print(f"Feature count: {len(train_loader.dataset.feature_names)}")
    print(f"Scaler min/max: {scaler.min_:.3f}, {scaler.max_:.3f}")


if __name__ == "__main__":
    _demo()
