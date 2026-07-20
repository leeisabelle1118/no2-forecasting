"""AirNow NO2 data loader for time-series model training.

The AirNow dataset (/mnt/data3/AirNow/) contains one NetCDF-4 file per day
with 24 hourly observations across 182 ground monitoring sites:

    Dimensions:  time (24 h), site (182)
    Variables:   no2 (PPB), aqi, latitude, longitude, site_name, agency
    Date range:  2023-07-01 → 2024-09-30  (459 days)

Key functions
-------------
load_all()          Load every day into a single (time, site) DataFrame indexed
                    by UTC hour and with one column per site code. Good for
                    quick exploration and Transformer input.

load_sequences()    Slide a window of length `seq_len` across the time axis and
                    return (X, y, timestamps) tensors ready for model training.

site_meta()         Return a DataFrame of site lat/lon/name for mapping.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime

import numpy as np
import netCDF4 as nc
import pandas as pd

DATA_DIR = "/mnt/data3/AirNow"

# ── Canonical time-series split boundaries (UTC, inclusive ends) ───────────────
# Dataset: 2023-07-01 → 2024-09-30  (15 months, 10 992 h)
# Full 12-month training window (split into train-proper & validation):
# Train-proper : 2023-07-01 → 2024-05-31  (11 months) ┐
# Validation   : 2024-06-01 → 2024-06-30  ( 1 month ) ┤ = 12 months (training data)
# Test        : 2024-07-01 → 2024-09-30  ( 3 months) ┘
#
# Windows are assigned by *start* timestamp — chronological, leak-free.
TRAIN_END       = pd.Timestamp("2024-05-31 23:00")   # last training-proper window start-hour
FULL_TRAIN_END  = pd.Timestamp("2024-06-30 23:00")   # last validation window start-hour (end of 12-month training)


def get_train_mean(data_dir: str = DATA_DIR, train_end: str | None = None) -> np.ndarray:
    """Compute the mean NO2 during the entire 12-month training period.
    
    By default, computes mean over the full 12-month training window (training-proper + validation).
    This ensures both training and validation data are normalized by the same reference.
    
    Parameters
    ----------
    data_dir  : path to AirNow NetCDF folder
    train_end : timestamp string (e.g. "2024-06-30 23:00") marking the last training hour
                (defaults to FULL_TRAIN_END to include full 12-month training window)
    
    Returns
    -------
    mean : float32 array of shape (n_sites,) with per-site mean from training period
    """
    if train_end is None:
        train_end_ts = FULL_TRAIN_END  # default: full 12-month training period
    else:
        train_end_ts = pd.Timestamp(train_end)
    
    df = load_all(data_dir).clip(lower=0.0)
    df_train = df[df.index <= train_end_ts]
    # Use nanmean to ignore NaN values, then fill remaining NaNs with 1.0
    mean = np.nanmean(df_train.values.astype("float32"), axis=0)
    mean = np.where(np.isnan(mean), 1.0, mean)  # avoid /0 and NaN
    return mean


def _nc_files(data_dir: str = DATA_DIR):
    """Return sorted list of AirNow NetCDF paths."""
    return sorted(glob.glob(os.path.join(data_dir, "airnow_no2_*.nc")))


def site_meta(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Return a DataFrame[site_code, lat, lon, name, agency] from the first file."""
    f = _nc_files(data_dir)[0]
    with nc.Dataset(f) as ds:
        codes = np.array(ds["site"][:], dtype=str)
        lats  = ds["latitude"][:].filled(np.nan)
        lons  = ds["longitude"][:].filled(np.nan)
        names = np.array(ds["site_name"][:], dtype=str)
        agencies = np.array(ds["agency"][:], dtype=str)
    return pd.DataFrame({"site": codes, "lat": lats, "lon": lons,
                         "name": names, "agency": agencies}).set_index("site")


def load_all(data_dir: str = DATA_DIR,
             variable: str = "no2") -> pd.DataFrame:
    """Load all files into a (n_hours × n_sites) DataFrame.

    Index: UTC datetime at hourly frequency.
    Columns: site codes.
    Values: NO₂ in PPB (NaN for missing / below-detection observations).
    """
    files = _nc_files(data_dir)
    frames = []
    for path in files:
        with nc.Dataset(path) as ds:
            # time in hours since the file's date
            t_units = ds["time"].units          # e.g. "hours since 2023-07-01 00:00:00"
            t_vals  = ds["time"][:].filled(0)
            times   = nc.num2date(t_vals, t_units, calendar="proleptic_gregorian")
            times   = [datetime(t.year, t.month, t.day, t.hour) for t in times]

            codes = np.array(ds["site"][:], dtype=str)
            data  = ds[variable][:].filled(np.nan)  # shape (24, 182)

        df = pd.DataFrame(data, index=pd.DatetimeIndex(times), columns=codes)
        frames.append(df)

    result = pd.concat(frames).sort_index()
    # Clamp physically impossible negatives
    result = result.clip(lower=0.0)
    return result


def load_sequences(data_dir: str = DATA_DIR,
                   variable: str = "no2",
                   seq_len: int = 24,
                   pred_len: int = 1,
                   stride: int = 1,
                   fill_nan: float = 0.0,
                   normalize: bool = True,
                   norm_end: str | None = None):
    """Return sliding-window sequences ready for Transformer / Mamba training.

    Parameters
    ----------
    seq_len   : look-back window (hours).
    pred_len  : forecast horizon (hours).
    stride    : step between consecutive windows.
    fill_nan  : value to substitute for missing observations.
    normalize : if True, divide each site by its mean so values are O(1).
    norm_end  : timestamp string (e.g. "2024-04-30 23:00") marking the last
                hour to include when computing the normalisation mean — use
                TRAIN_END to keep normalisation strictly inside training data.
                If None, falls back to the first 80% of timesteps.

    Returns
    -------
    X          : float32 array of shape (n_samples, seq_len, n_sites)
    y          : float32 array of shape (n_samples, pred_len, n_sites)
    timestamps : list of pd.Timestamp marking the *start* of each X window
    site_codes : list of site-code strings (feature names)
    """
    df = load_all(data_dir, variable).fillna(fill_nan).clip(lower=0.0)
    arr = df.values.astype("float32")           # (n_hours, n_sites)

    if normalize:
        if norm_end is not None:
            norm_mask = df.index <= pd.Timestamp(norm_end)
        else:
            norm_mask = np.zeros(len(arr), dtype=bool)
            norm_mask[: int(len(arr) * 0.8)] = True   # backward-compat fallback
        mean = arr[norm_mask].mean(axis=0, keepdims=True)
        mean = np.where(mean == 0, 1.0, mean)           # avoid /0
        arr = arr / mean

    total = seq_len + pred_len
    X_list, y_list, ts_list = [], [], []
    for i in range(0, len(arr) - total + 1, stride):
        X_list.append(arr[i : i + seq_len])
        y_list.append(arr[i + seq_len : i + total])
        ts_list.append(df.index[i])

    X = np.stack(X_list)   # (n, seq_len, n_sites)
    y = np.stack(y_list)   # (n, pred_len, n_sites)
    return X, y, ts_list, list(df.columns)


if __name__ == "__main__":
    print("Site metadata (first 5):")
    print(site_meta().head())

    print("\nLoading all files ...")
    df = load_all()
    print(f"Shape: {df.shape}  |  Date range: {df.index[0]} → {df.index[-1]}")
    print(f"Missing %: {df.isna().mean().mean():.1%}")
    print(f"NO2 range: {df.min().min():.2f} – {df.max().max():.2f} PPB")
