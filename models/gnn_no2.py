"""GNN baseline for NO₂ hourly time-series forecasting.

Architecture
------------
1. **Spatial GCN** — at every time step, k-NN Graph Convolutional layers
   propagate information between neighbouring monitoring sites.
2. **Weight-shared GRU** — each site's spatially-enriched feature sequence
   is passed through a GRU (weights shared across all sites) to model the
   temporal dynamics.
3. **Per-site linear head** — maps each site's GRU hidden state to the
   forecast horizon.

The k-NN graph is built from station lat/lon coordinates (using great-circle
distance) and stored as a non-trainable buffer so it is saved with the
checkpoint and never needs to be recomputed at inference time.

    Input  : (batch, seq_len,  n_sites)  — normalised NO₂ look-back window
    Output : (batch, pred_len, n_sites)  — forecast horizon

Usage
-----
    from data.load_airnow import site_meta
    from models.gnn_no2 import build_knn_adj, NO2GNN

    meta = site_meta()
    adj  = build_knn_adj(meta["lat"].values, meta["lon"].values, k=5)
    model = NO2GNN(n_sites=len(meta), seq_len=24, pred_len=6, adj=adj)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_adj(
    lats: np.ndarray,
    lons: np.ndarray,
    k: int = 5,
) -> torch.Tensor:
    """Return a normalised adjacency tensor from station coordinates.

    Builds a symmetric k-NN graph connecting each station to its k nearest
    neighbours by great-circle distance, adds self-loops, then applies
    symmetric normalisation  D⁻½ A D⁻½.

    Parameters
    ----------
    lats, lons : 1-D float arrays of length n_sites (degrees).
    k          : number of nearest neighbours per station (default 5).

    Returns
    -------
    adj_norm : float32 tensor of shape (n_sites, n_sites).
    """
    n = len(lats)
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)

    # Vectorised Haversine distance matrix
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2)
    dist = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # (n, n) in radians

    # k-NN adjacency (exclude self, which has dist=0)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nn_idx = np.argsort(dist[i])[1 : k + 1]
        adj[i, nn_idx] = 1.0

    # Symmetrise + self-loops
    adj = np.maximum(adj, adj.T)
    adj += np.eye(n, dtype=np.float32)

    # Symmetric normalisation: D⁻½ A D⁻½
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    adj_norm = (d_inv_sqrt[:, None] * adj) * d_inv_sqrt[None, :]

    return torch.from_numpy(adj_norm)


# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """One graph convolutional layer: h = ReLU( LayerNorm( Â X W + b ) ).

    Parameters
    ----------
    in_features, out_features : feature dimensions per node.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=True)
        self.norm   = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x   : (batch, n_sites, in_features)
        # adj : (n_sites, n_sites) — normalised adjacency (on same device)
        h = self.weight(x)                           # (B, S, out_features)
        h = torch.einsum("ij,bjf->bif", adj, h)     # graph mixing: Â h
        return F.relu(self.norm(h))


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class NO2GNN(nn.Module):
    """k-NN spatial GCN + weight-shared GRU for multi-site NO₂ forecasting.

    Parameters
    ----------
    n_sites   : number of monitoring sites (input/output features).
    seq_len   : look-back window length (hours).
    pred_len  : forecast horizon (hours).
    d_model   : GCN output and GRU hidden dimension.
    n_layers  : number of GCN layers applied at each time step (default 2).
    k_nn      : k used when building the graph (stored for reference only).
    adj       : pre-built normalised adjacency tensor of shape (n_sites, n_sites).
                Stored as a buffer so it is saved with the checkpoint.
                Falls back to identity (no graph mixing) if None.
    dropout   : dropout rate applied after the GRU.
    """
    def __init__(
        self,
        n_sites: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_layers: int = 2,
        k_nn: int = 5,
        adj: torch.Tensor | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sites  = n_sites
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.k_nn     = k_nn

        # Adjacency stored as a non-trainable buffer (saved with checkpoint)
        self.register_buffer(
            "adj",
            adj if adj is not None else torch.eye(n_sites),
        )

        # GCN stack: scalar NO₂ value (1-D) → d_model per node
        gcn_dims = [1] + [d_model] * n_layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(gcn_dims[i], gcn_dims[i + 1]) for i in range(n_layers)
        ])
        # Skip-connection projections (handles dim mismatch on first layer)
        self.gcn_skip = nn.ModuleList([
            nn.Linear(gcn_dims[i], gcn_dims[i + 1], bias=False)
            for i in range(n_layers)
        ])

        # Temporal GRU — weight-shared across all sites
        self.gru  = nn.GRU(input_size=d_model, hidden_size=d_model,
                           num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)

        # Per-site output head: d_model → pred_len (one step per site)
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, n_sites)
        B, L, S = x.shape

        # ── Spatial: GCN at every time step ───────────────────────────────────
        # (B, L, S) → (B*L, S, 1): treat each time step as a batch of graphs
        h = x.reshape(B * L, S, 1)
        for gcn, skip in zip(self.gcn_layers, self.gcn_skip):
            h = gcn(h, self.adj) + skip(h)   # residual skip connection

        # h : (B*L, S, d_model)
        d = h.shape[-1]

        # ── Temporal: weight-shared GRU per site ──────────────────────────────
        # Reshape to (B*S, L, d_model) so the GRU processes each site's
        # time series independently (weights are shared across sites)
        h = h.reshape(B, L, S, d)      # (B, L, S, d_model)
        h = h.permute(0, 2, 1, 3)      # (B, S, L, d_model)
        h = h.reshape(B * S, L, d)     # (B*S, L, d_model)

        _, h_n = self.gru(h)           # h_n : (1, B*S, d_model)
        h = self.drop(h_n.squeeze(0))  # (B*S, d_model)

        # ── Per-site forecast ──────────────────────────────────────────────────
        out = self.output_proj(h)                    # (B*S, pred_len)
        out = out.reshape(B, S, self.pred_len)        # (B, S, pred_len)
        return out.permute(0, 2, 1)                  # (B, pred_len, S) ✓


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers (mirror transformer_no2.py interface)
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


def evaluate(
    model: NO2GNN,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> tuple[float, float]:
    """Return (MSE, MAE) on the given arrays."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NO2GNN — quick forward-pass check")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lats = np.linspace(33.5, 48.5, 10)
    lons = np.linspace(-124.0, -104.0, 10)
    adj  = build_knn_adj(lats, lons, k=3)

    m = NO2GNN(n_sites=10, seq_len=24, pred_len=6, d_model=32, adj=adj).to(device)
    x = torch.randn(4, 24, 10).to(device)
    out = m(x)
    assert out.shape == (4, 6, 10), f"Unexpected shape: {out.shape}"
    params = sum(p.numel() for p in m.parameters())
    print(f"  output shape : {tuple(out.shape)}  ✓")
    print(f"  parameters   : {params:,}")
    print(f"  device       : {device}")
    print(f"  adj shape    : {tuple(adj.shape)}")
    print(f"  adj non-zero : {(adj > 0).sum().item()}")
