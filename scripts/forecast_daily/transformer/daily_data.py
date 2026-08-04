"""PyTorch Dataset/DataLoader for univariate AirNow NO2 forecasting.

Requirements implemented:
- Input DataFrame has columns: 'date', 'airnow_no2'
- Lookback window K=7 days
- Forecast lead time H=1 day (predict t+1)
- X shape per batch: (batch_size, 7, 1)
- y shape per batch: (batch_size, 1)
- Chronological train/test split (no random shuffle)
- Min-max scaling fit on TRAIN ONLY
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


LOOKBACK_DAYS = 7
LEAD_DAYS = 1


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
            # Constant training signal -> map everything to 0.0
            return np.zeros_like(x, dtype=np.float32)
        return ((x - self.min_) / denom).astype(np.float32)

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        return (x_scaled * (self.max_ - self.min_) + self.min_).astype(np.float32)


class AirNowNO2Dataset(Dataset):
    """Windowed univariate dataset for forecasting y(t+H) from past K points."""

    def __init__(self, values_scaled: np.ndarray, lookback: int = LOOKBACK_DAYS, lead: int = LEAD_DAYS):
        """
        Args:
            values_scaled: 1D scaled NO2 array in chronological order.
            lookback: Number of past timesteps K.
            lead: Forecast lead H, where target is at t+H.
        """
        if values_scaled.ndim != 1:
            raise ValueError("values_scaled must be a 1D array")
        if len(values_scaled) < lookback + lead:
            raise ValueError("Not enough data points for requested lookback/lead")

        self.lookback = lookback
        self.lead = lead

        X, y = self._make_windows(values_scaled, lookback, lead)
        # X: (N, K) -> (N, K, 1), y: (N,) -> (N, 1)
        self.X = torch.from_numpy(X[:, :, None]).float()
        self.y = torch.from_numpy(y[:, None]).float()

    @staticmethod
    def _make_windows(values: np.ndarray, lookback: int, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        # End index of lookback window is i-1, target at i+lead-1
        # Here i is window start.
        max_start = len(values) - (lookback + lead) + 1
        for i in range(max_start):
            x_window = values[i : i + lookback]
            y_target = values[i + lookback + lead - 1]
            X_list.append(x_window)
            y_list.append(y_target)

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
        return X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def prepare_series(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and enforce chronological order."""
    required = {"date", "airnow_no2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    out = df[["date", "airnow_no2"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").dropna(subset=["airnow_no2"]).reset_index(drop=True)
    return out


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological split: first train_ratio for train, rest for test."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    split_idx = int(len(df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Split produced empty train or test set")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def make_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    lookback: int = LOOKBACK_DAYS,
    lead: int = LEAD_DAYS,
) -> Tuple[DataLoader, DataLoader, MinMaxScaler1D]:
    """Build train/test dataloaders with train-only min-max scaling."""
    df = prepare_series(df)
    train_df, test_df = chronological_split(df, train_ratio=train_ratio)

    train_values = train_df["airnow_no2"].to_numpy(dtype=np.float32)
    test_values = test_df["airnow_no2"].to_numpy(dtype=np.float32)

    # Fit scaler ONLY on training data to avoid leakage.
    scaler = MinMaxScaler1D.fit(train_values)
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)

    train_ds = AirNowNO2Dataset(train_scaled, lookback=lookback, lead=lead)
    test_ds = AirNowNO2Dataset(test_scaled, lookback=lookback, lead=lead)

    # Keep chronological order by setting shuffle=False.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


def _demo() -> None:
    """Small runnable demo with synthetic daily data."""
    n_days = 120
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    no2 = 20 + 5 * np.sin(np.arange(n_days) * 2 * np.pi / 7) + np.random.normal(0, 0.8, n_days)

    df = pd.DataFrame({"date": dates, "airnow_no2": no2})

    train_loader, test_loader, scaler = make_dataloaders(
        df,
        batch_size=16,
        train_ratio=0.8,
        lookback=LOOKBACK_DAYS,
        lead=LEAD_DAYS,
    )

    xb, yb = next(iter(train_loader))
    print(f"Train batch X shape: {tuple(xb.shape)}")  # (batch, 7, 1)
    print(f"Train batch y shape: {tuple(yb.shape)}")  # (batch, 1)
    print(f"Scaler min/max: {scaler.min_:.3f}, {scaler.max_:.3f}")


if __name__ == "__main__":
    _demo()
