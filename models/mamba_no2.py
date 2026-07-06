"""Mamba (SSM) model for NO2 hourly time-series forecasting.

Mamba (Gu & Dao, 2023) is a selective state-space model that processes
sequences in O(n) time — faster than Transformer on long sequences and
often better at capturing temporal patterns in environmental time-series.

This module provides a lightweight, pure-PyTorch Mamba block that does not
require the CUDA-fused ``mamba_ssm`` kernels, so it runs on CPU too. If you
have a CUDA GPU and want the full optimised implementation:

    pip install mamba-ssm causal-conv1d

Architecture
------------
Input  : (batch, seq_len,  n_sites)  — look-back window
Output : (batch, pred_len, n_sites)  — forecast horizon

Usage
-----
    from models.mamba_no2 import NO2Mamba, train, evaluate
    model = NO2Mamba(n_sites=182, seq_len=24, pred_len=6)
    train(model, X_train, y_train, X_val, y_val)
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba block (no custom CUDA kernels required)
# Based on: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
# State Spaces" (2023)  https://arxiv.org/abs/2312.00752
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single Mamba SSM block.

    Parameters
    ----------
    d_model   : input / output dimension.
    d_state   : SSM state dimension (N in the paper).
    d_conv    : local depthwise-conv width.
    expand    : inner dimension = expand × d_model.
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner = expand * d_model
        self.d_state = d_state

        # Input projection (x and z branches)
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        # Short depthwise conv on x
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=d_inner, bias=True)
        # SSM parameter projections
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + d_state, bias=False)
        self.dt_proj  = nn.Linear(d_state, d_inner, bias=True)

        # Initialise A as negative log of 1..N (standard HiPPO init)
        A = torch.arange(1, d_state + 1).float().log().unsqueeze(0).expand(d_inner, -1)
        self.A_log    = nn.Parameter(A)
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Discretised SSM scan (simplified sequential implementation)."""
        B, L, d = x.shape                          # batch, seq, d_inner
        A = -self.A_log.exp()                       # (d_inner, d_state)

        # B_hat, C_hat, dt from x
        bcd = self.x_proj(x)                       # (B, L, 2*d_state + d_state)
        split = self.d_state
        B_hat = bcd[..., :split]                   # (B, L, d_state)
        C_hat = bcd[..., split:2*split]
        dt    = F.softplus(self.dt_proj(bcd[..., 2*split:]))  # (B, L, d_inner)

        # Discretise: A_bar = exp(dt * A), B_bar = dt * B_hat
        # For efficiency: parallel scan via simple cumsum approximation
        # (exact for the purposes of learning; production use should use the
        #  associative scan from the mamba_ssm CUDA kernel)
        h = torch.zeros(B, d, split, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t   = dt[:, t, :].unsqueeze(-1)    # (B, d_inner, 1)
            A_bar  = torch.exp(dt_t * A.unsqueeze(0))          # (B, d_inner, d_state)
            B_bar  = dt_t * B_hat[:, t, :].unsqueeze(1)        # (B, d_inner, d_state)
            h      = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)
            y_t    = (h * C_hat[:, t, :].unsqueeze(1)).sum(-1) # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                             # (B, L, d_inner)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)
        xz  = self.in_proj(x)                        # (B, L, 2*d_inner)
        x_b, z = xz.split(self.d_inner, dim=-1)

        # Depthwise conv along sequence
        x_b = self.conv1d(x_b.transpose(1, 2)).transpose(1, 2)[:, :x.size(1), :]
        x_b = F.silu(x_b)

        y = self.ssm(x_b)
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class NO2Mamba(nn.Module):
    """Stacked Mamba SSM blocks for multi-site NO2 forecasting.

    Parameters
    ----------
    n_sites   : number of monitoring sites (input/output features).
    seq_len   : look-back window length (hours).
    pred_len  : forecast horizon (hours).
    d_model   : internal state dimension.
    n_layers  : number of Mamba blocks.
    d_state   : SSM latent state size per block.
    expand    : inner-dim expansion factor inside each block.
    dropout   : dropout rate between blocks.
    """
    def __init__(
        self,
        n_sites: int = 182,
        seq_len: int = 24,
        pred_len: int = 1,
        d_model: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.n_sites  = n_sites

        self.input_proj = nn.Linear(n_sites, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model * seq_len, pred_len * n_sites)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_sites)
        h = self.input_proj(x)              # (B, seq_len, d_model)
        for block in self.blocks:
            h = self.drop(block(h))
        h = h.flatten(1)                    # (B, seq_len * d_model)
        out = self.output_proj(h)           # (B, pred_len * n_sites)
        return out.view(-1, self.pred_len, self.n_sites)


# Reuse the same train / evaluate helpers as the Transformer module
from models.transformer_no2 import (   # noqa: E402
    _make_loader, train, evaluate
)

__all__ = ["NO2Mamba", "MambaBlock", "train", "evaluate"]
