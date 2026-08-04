"""Simple daily lagged dataset utilities for forecast_daily.

Pipeline contract:
- One daily NO2 value per day
- Lookback window K=7 days
- Direct one-step horizon H=1 day (t+1)
- Chronological train/test split
- Min-max scaling fit on train only
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
            # Constant training signal -> map everything to 0.0
            return np.zeros_like(x, dtype=np.float32)
        return ((x - self.min_) / denom).astype(np.float32)

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        return (x_scaled * (self.max_ - self.min_) + self.min_).astype(np.float32)


class AirNowNO2Dataset(Dataset):
    """Windowed univariate dataset for forecasting y(t+H) from past K values."""

    def __init__(
        self,
        values_scaled: np.ndarray,
        lookback: int = LOOKBACK_DAYS,
        lead: int = LEAD_DAYS,
        dates: np.ndarray | pd.Series | None = None,
    ):
        """
        Args:
            values_scaled: 1D scaled NO2 array in chronological order.
            lookback: Number of past timesteps K.
            lead: Forecast lead H, where target is at t+H.
        """
        if values_scaled.ndim != 1:
            raise ValueError("values_scaled must be a 1D array")
        if lead != LEAD_DAYS:
            raise ValueError(
                f"forecast_daily supports direct one-step forecasting only (lead={LEAD_DAYS}); got lead={lead}"
            )
        if len(values_scaled) < lookback + lead:
            raise ValueError("Not enough data points for requested lookback/lead")
        if dates is not None and len(dates) != len(values_scaled):
            raise ValueError("dates must have the same length as values_scaled")

        self.lookback = lookback
        self.lead = lead

        X, y, target_dates = self._make_windows(values_scaled, lookback, lead, dates)
        # X: (N, K, 1), y: (N,) -> (N, 1)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y[:, None]).float()
        self.target_dates = pd.to_datetime(target_dates) if target_dates is not None else None

    @staticmethod
    def _make_windows(
        values: np.ndarray,
        lookback: int,
        lead: int,
        dates: np.ndarray | pd.Series | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        X_list, y_list, date_list = [], [], []
        # End index of lookback window is i-1, target at i+lead-1
        # Here i is window start.
        max_start = len(values) - (lookback + lead) + 1
        for i in range(max_start):
            x_window = values[i : i + lookback][:, None]
            y_target = values[i + lookback + lead - 1]
            X_list.append(x_window)
            y_list.append(y_target)
            if dates is not None:
                date_list.append(pd.Timestamp(dates[i + lookback + lead - 1]))

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
        date_arr = np.asarray(date_list, dtype="datetime64[ns]") if dates is not None else None
        return X, y, date_arr

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

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").dropna(subset=["airnow_no2"]).reset_index(drop=True)
    return out


def resolve_train_end(df: pd.DataFrame, train_end: str | pd.Timestamp | None = "auto") -> pd.Timestamp:
    """Resolve train-end boundary, supporting an automatic full-year cutoff.

    When ``train_end`` is ``None`` or ``"auto"``, this uses the first full
    calendar year from the earliest available daily timestamp and requires at
    least one later day for test evaluation.
    """
    if train_end is None or (isinstance(train_end, str) and train_end.lower() == "auto"):
        min_date = pd.Timestamp(df["date"].min())
        max_date = pd.Timestamp(df["date"].max())

        # First full calendar year from dataset start (inclusive end date).
        inferred = min_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        if inferred >= max_date:
            raise ValueError(
                "Unable to infer a full-year train/test split from daily data. "
                "Need at least 1 year of daily data plus 1 extra day for test; "
                f"found range {min_date.date()} to {max_date.date()}."
            )
        return inferred

    return pd.Timestamp(train_end)


def chronological_split(df: pd.DataFrame, train_end: str | pd.Timestamp | None = "auto") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological split using an explicit or automatic full-year boundary."""
    train_end_ts = resolve_train_end(df, train_end=train_end)

    train_df = df[df["date"] <= train_end_ts].copy()
    test_df = df[df["date"] > train_end_ts].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Full-year split produced an empty train or test set. "
            f"train_end={train_end_ts}, rows={len(df)}"
        )
    return train_df, test_df


def make_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    train_end: str | pd.Timestamp | None = "auto",
    lookback: int = LOOKBACK_DAYS,
    lead: int = LEAD_DAYS,
) -> Tuple[DataLoader, DataLoader, MinMaxScaler1D]:
    """Build train/test dataloaders for simple daily lagged forecasting."""
    if lead != LEAD_DAYS:
        raise ValueError(
            f"forecast_daily supports direct one-step forecasting only (lead={LEAD_DAYS}); got lead={lead}"
        )

    df = prepare_series(df)
    train_df, test_df = chronological_split(df, train_end=train_end)

    # Guardrails: ensure strict chronological train->test ordering.
    if not train_df["date"].is_monotonic_increasing or not test_df["date"].is_monotonic_increasing:
        raise ValueError("Train/test splits must be time-ordered by date")
    if pd.Timestamp(train_df["date"].max()) >= pd.Timestamp(test_df["date"].min()):
        raise ValueError(
            "Train/test split is not strictly chronological: "
            f"train_max={pd.Timestamp(train_df['date'].max())}, "
            f"test_min={pd.Timestamp(test_df['date'].min())}"
        )

    train_values = train_df["airnow_no2"].to_numpy(dtype=np.float32)
    test_values = test_df["airnow_no2"].to_numpy(dtype=np.float32)

    train_dates = train_df["date"].to_numpy()
    test_dates = test_df["date"].to_numpy()

    # Fit scaler ONLY on training data to avoid leakage.
    scaler = MinMaxScaler1D.fit(train_values)
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)

    train_ds = AirNowNO2Dataset(
        train_scaled,
        lookback=lookback,
        lead=lead,
        dates=train_dates,
    )
    test_ds = AirNowNO2Dataset(
        test_scaled,
        lookback=lookback,
        lead=lead,
        dates=test_dates,
    )

    # Keep chronological order by setting shuffle=False.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Attach metadata for downstream scripts and checks.
    train_ds.feature_names = ["airnow_no2_lagged"]
    test_ds.feature_names = ["airnow_no2_lagged"]
    train_ds.split_train_end = pd.Timestamp(train_df["date"].max())
    test_ds.split_test_start = pd.Timestamp(test_df["date"].min())
    train_ds.forecast_horizon_days = LEAD_DAYS
    test_ds.forecast_horizon_days = LEAD_DAYS
    train_ds.forecast_mode = "direct_one_step_t_plus_1"
    test_ds.forecast_mode = "direct_one_step_t_plus_1"

    if train_ds.y.shape[1] != 1 or test_ds.y.shape[1] != 1:
        raise ValueError("Direct one-step setup must produce scalar daily targets with shape (N, 1)")

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
        train_end=FULL_YEAR_END,
        lookback=LOOKBACK_DAYS,
        lead=LEAD_DAYS,
    )

    xb, yb = next(iter(train_loader))
    print(f"Train batch X shape: {tuple(xb.shape)}")  # (batch, 7, 1)
    print(f"Train batch y shape: {tuple(yb.shape)}")  # (batch, 1)
    print(f"Feature names: {train_loader.dataset.feature_names}")
    print(f"Scaler min/max: {scaler.min_:.3f}, {scaler.max_:.3f}")


if __name__ == "__main__":
    _demo()
