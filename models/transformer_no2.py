"""Baseline Transformer model for NO2 hourly time-series forecasting.

Architecture
------------
A standard encoder-only Transformer that ingests a (seq_len, n_sites) window
of past hourly NO₂ observations and predicts the next pred_len hours across
all sites.

    Input  : (batch, seq_len,  n_sites)  — look-back window
    Output : (batch, pred_len, n_sites)  — forecast horizon

Usage
-----
    from models.transformer_no2 import NO2Transformer, train, evaluate
    model = NO2Transformer(n_sites=182, seq_len=24, pred_len=6)
    train(model, X_train, y_train)
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class NO2Transformer(nn.Module):
    """Encoder-only Transformer for multi-site NO2 forecasting.

    Parameters
    ----------
    n_sites   : number of monitoring sites (input/output features).
    seq_len   : look-back window length (hours).
    pred_len  : forecast horizon (hours).
    d_model   : internal embedding dimension.
    n_heads   : number of self-attention heads (must divide d_model).
    n_layers  : number of Transformer encoder layers.
    d_ff      : feed-forward hidden dimension.
    dropout   : dropout rate.
    """
    def __init__(
        self,
        n_sites: int = 182,
        seq_len: int = 24,
        pred_len: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.n_sites  = n_sites

        # Project raw site values into the model dimension
        self.input_proj = nn.Linear(n_sites, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project last seq_len encoded positions to pred_len × n_sites
        self.output_proj = nn.Linear(d_model * seq_len, pred_len * n_sites)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_sites)
        h = self.pos_enc(self.input_proj(x))   # (batch, seq_len, d_model)
        h = self.encoder(h)                    # (batch, seq_len, d_model)
        h = h.flatten(1)                       # (batch, seq_len * d_model)
        out = self.output_proj(h)              # (batch, pred_len * n_sites)
        return out.view(-1, self.pred_len, self.n_sites)


def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


def train(
    model: NO2Transformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 5,
    device: str | None = None,
) -> list[dict]:
    """Train the model and return a list of per-epoch metric dicts."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader  = _make_loader(X_train, y_train, batch_size, shuffle=True)
    history, best_val, stale = [], float("inf"), 0

    for epoch in range(1, epochs + 1):
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

        row = {"epoch": epoch, "train_mse": train_loss}
        if X_val is not None:
            val_mse, val_mae = evaluate(model, X_val, y_val,
                                        batch_size=batch_size, device=device)
            row.update({"val_mse": val_mse, "val_mae": val_mae})
            print(f"Epoch {epoch:3d} | train MSE={train_loss:.4f} "
                  f"val MSE={val_mse:.4f} MAE={val_mae:.4f}")
            if val_mse < best_val:
                best_val, stale = val_mse, 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    history.append(row)
                    break
        else:
            print(f"Epoch {epoch:3d} | train MSE={train_loss:.4f}")
        history.append(row)
    return history


def evaluate(
    model: NO2Transformer,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> tuple[float, float]:
    """Return (MSE, MAE) on a held-out set."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    loader = _make_loader(X, y, batch_size, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            targets.append(yb.numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    return mse, mae
