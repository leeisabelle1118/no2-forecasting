"""Canonical daily dataset contract for forecast_daily baseline.

Contract (enforced):
- Input schema: columns [date, airnow_no2]
- One scalar daily NO2 value per row
- Lookback K=7 days
- Horizon H=1 day (direct t+1)
- Strict chronological split (train <= train_end < test)
- Train-only min-max scaling
- Window tensors: X shape (N, 7, 1), y shape (N, 1)
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
    def __init__(
        self,
        values_scaled: np.ndarray,
        lookback: int = LOOKBACK_DAYS,
        lead: int = LEAD_DAYS,
        dates: np.ndarray | pd.Series | None = None,
    ):
        if values_scaled.ndim != 1:
            raise ValueError("values_scaled must be a 1D array")
        if lookback != LOOKBACK_DAYS:
            raise ValueError(
                f"forecast_daily baseline enforces lookback={LOOKBACK_DAYS}; got lookback={lookback}"
            )
        if lead != LEAD_DAYS:
            raise ValueError(
                f"forecast_daily baseline enforces lead={LEAD_DAYS} (direct t+1); got lead={lead}"
            )
        if len(values_scaled) < lookback + lead:
            raise ValueError("Not enough data points for requested lookback/lead")
        if dates is not None and len(dates) != len(values_scaled):
            raise ValueError("dates must have the same length as values_scaled")

        self.lookback = lookback
        self.lead = lead

        X, y, target_dates = self._make_windows(values_scaled, lookback, lead, dates)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y[:, None]).float()
        self.target_dates = pd.to_datetime(target_dates) if target_dates is not None else None

        if self.X.ndim != 3 or self.X.shape[1] != LOOKBACK_DAYS or self.X.shape[2] != 1:
            raise ValueError(
                "Dataset contract violation: expected X shape (N, 7, 1); "
                f"got {tuple(self.X.shape)}"
            )
        if self.y.ndim != 2 or self.y.shape[1] != 1:
            raise ValueError(
                "Dataset contract violation: expected y shape (N, 1); "
                f"got {tuple(self.y.shape)}"
            )

    @staticmethod
    def _make_windows(
        values: np.ndarray,
        lookback: int,
        lead: int,
        dates: np.ndarray | pd.Series | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        X_list, y_list, date_list = [], [], []
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
    required = {"date", "airnow_no2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["airnow_no2"] = pd.to_numeric(out["airnow_no2"], errors="coerce")
    out = out.sort_values("date").dropna(subset=["airnow_no2"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Daily dataset is empty after cleaning")
    if out["date"].duplicated().any():
        dup = out.loc[out["date"].duplicated(), "date"].iloc[0]
        raise ValueError(f"Duplicate date found in daily dataset: {pd.Timestamp(dup).date()}")
    return out


def resolve_train_end(df: pd.DataFrame, train_end: str | pd.Timestamp | None = "auto") -> pd.Timestamp:
    if train_end is None or (isinstance(train_end, str) and train_end.lower() == "auto"):
        min_date = pd.Timestamp(df["date"].min())
        max_date = pd.Timestamp(df["date"].max())
        inferred = min_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        if inferred >= max_date:
            raise ValueError(
                "Unable to infer full-year split. Need >= 1 year train plus >= 1 day test; "
                f"found {min_date.date()} to {max_date.date()}."
            )
        return inferred

    return pd.Timestamp(train_end)


def chronological_split(
    df: pd.DataFrame, train_end: str | pd.Timestamp | None = "auto"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_end_ts = resolve_train_end(df, train_end=train_end)

    train_df = df[df["date"] <= train_end_ts].copy()
    test_df = df[df["date"] > train_end_ts].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Chronological split produced empty train or test set: "
            f"train_rows={len(train_df)} test_rows={len(test_df)} train_end={train_end_ts.date()}"
        )

    if pd.Timestamp(train_df["date"].max()) >= pd.Timestamp(test_df["date"].min()):
        raise ValueError(
            "Chronological split violation: "
            f"train_max={pd.Timestamp(train_df['date'].max())} "
            f"test_min={pd.Timestamp(test_df['date'].min())}"
        )

    return train_df, test_df


def make_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    train_end: str | pd.Timestamp | None = "auto",
    lookback: int = LOOKBACK_DAYS,
    lead: int = LEAD_DAYS,
) -> Tuple[DataLoader, DataLoader, MinMaxScaler1D]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")

    df = prepare_series(df)
    train_df, test_df = chronological_split(df, train_end=train_end)

    train_values = train_df["airnow_no2"].to_numpy(dtype=np.float32)
    test_values = test_df["airnow_no2"].to_numpy(dtype=np.float32)

    scaler = MinMaxScaler1D.fit(train_values)
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)

    train_ds = AirNowNO2Dataset(
        train_scaled,
        lookback=lookback,
        lead=lead,
        dates=train_df["date"].to_numpy(),
    )
    test_ds = AirNowNO2Dataset(
        test_scaled,
        lookback=lookback,
        lead=lead,
        dates=test_df["date"].to_numpy(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_ds.feature_names = ["airnow_no2_lagged"]
    test_ds.feature_names = ["airnow_no2_lagged"]
    train_ds.split_train_end = pd.Timestamp(train_df["date"].max())
    test_ds.split_test_start = pd.Timestamp(test_df["date"].min())
    train_ds.forecast_horizon_days = LEAD_DAYS
    test_ds.forecast_horizon_days = LEAD_DAYS
    train_ds.forecast_mode = "direct_one_step_t_plus_1"
    test_ds.forecast_mode = "direct_one_step_t_plus_1"

    if train_ds.target_dates is None or test_ds.target_dates is None:
        raise ValueError("Target-date metadata was not attached to datasets")

    return train_loader, test_loader, scaler
