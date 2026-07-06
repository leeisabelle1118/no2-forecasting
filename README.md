# NO2 Forecasting — Transformer & Mamba

Time-series forecasting of ground-level NO₂ concentrations using
**Transformer** and **Mamba (SSM)** models trained on AirNow hourly
observations across ~182–197 US monitoring sites.

---

## Dataset — AirNow NO₂

| Property | Value |
|---|---|
| Source | [AirNow API](https://www.airnowapi.org/) |
| Data path | `/mnt/data3/AirNow/airnow_no2_YYYYMMDD.nc` |
| Date range | 2023-07-01 → 2024-09-30 (459 days) |
| Sites | ~197 ground monitoring stations |
| Temporal resolution | Hourly (24 observations/day) |
| Variables | `no2` (PPB), `aqi`, `latitude`, `longitude`, `site_name` |
| Total timesteps | 10,992 hours |
| Missing data | ~13.7 % overall |
| Spatial bbox | −128→−100 °E, 31→52 °N (Pacific NW to Great Plains) |

---

## Project structure

```
NO2 Forecasting/
├── data/
│   ├── __init__.py
│   └── load_airnow.py        # load_all(), load_sequences(), site_meta()
├── models/
│   ├── __init__.py
│   ├── transformer_no2.py    # Encoder-only Transformer (Vaswani et al. 2017)
│   └── mamba_no2.py          # Mamba SSM (Gu & Dao 2023), pure-PyTorch
├── notebooks/
│   └── 01_explore_airnow.ipynb   # EDA: site map, time series, diurnal cycle,
│                                 #      missing data, model forward-pass check
├── outputs/                  # Checkpoints (.pt), training history (.json),
│                             # comparison plots (.png)  — gitignored
├── train.py                  # CLI training script (Transformer or Mamba)
├── compare.py                # Load checkpoints, compare MSE/MAE, save plots
├── environment.yml           # Conda environment spec
├── requirements.txt          # pip fallback
└── LICENSE                   # MIT
```

---

## Environment setup

```bash
# Create a new conda env from the spec
mamba env create -f environment.yml
conda activate no2-forecasting

# — OR — install into an existing env (e.g. the cartopy env)
mamba install -y -n cartopy -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision pytorch-cuda=12.1 pandas scikit-learn tqdm
```

Optional (CUDA-optimised Mamba scan — GPU only):
```bash
pip install mamba-ssm causal-conv1d
```

---

## Quick start

### Training (CLI)

```bash
# Train Transformer for 50 epochs (default)
python train.py --model transformer

# Train Mamba with a 48-hour look-back and 12-hour forecast
python train.py --model mamba --seq-len 48 --pred-len 12

# All options
python train.py --help
```

Checkpoints are saved to `outputs/<model>_s<seq>_p<pred>_d<dim>.pt`.
Training history (loss curves, test MSE/MAE) is saved as a JSON alongside.

### Compare models

```bash
python compare.py
```

Generates four files in `outputs/`:

| File | Contents |
|---|---|
| `comparison_results.json` | MSE, MAE, parameter counts |
| `comparison_curves.png` | Train / val loss curves |
| `comparison_scatter.png` | Predicted vs actual scatter |
| `site_mae_map.png` | Per-site MAE on a lat/lon map |

### Python API

```python
import sys; sys.path.insert(0, ".")   # run from NO2 Forecasting/

from data.load_airnow import load_sequences
from models.transformer_no2 import NO2Transformer, train, evaluate
from models.mamba_no2 import NO2Mamba

X, y, timestamps, sites = load_sequences(seq_len=24, pred_len=6)
n = len(X); n_train = int(n*0.70); n_val = int(n*0.15)
X_tr, y_tr = X[:n_train], y[:n_train]
X_v,  y_v  = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_te, y_te = X[n_train+n_val:], y[n_train+n_val:]

# Transformer
tf = NO2Transformer(n_sites=len(sites), seq_len=24, pred_len=6)
train(tf, X_tr, y_tr, X_v, y_v, epochs=50)
print("Transformer test:", evaluate(tf, X_te, y_te))   # (MSE, MAE)

# Mamba
mb = NO2Mamba(n_sites=len(sites), seq_len=24, pred_len=6)
train(mb, X_tr, y_tr, X_v, y_v, epochs=50)
print("Mamba test:      ", evaluate(mb, X_te, y_te))
```

---

## Models

### Transformer (`models/transformer_no2.py`)

```
Input (B, seq_len, n_sites)
  → Linear(n_sites, d_model)
  → SinusoidalPositionalEncoding
  → n_layers × TransformerEncoderLayer(d_model, n_heads, d_ff)
  → Flatten
  → Linear(d_model·seq_len, pred_len·n_sites)
Output (B, pred_len, n_sites)
```

| Hyperparameter | Default |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `d_ff` | 256 |
| `dropout` | 0.1 |

### Mamba (`models/mamba_no2.py`)

```
Input (B, seq_len, n_sites)
  → Linear(n_sites, d_model)
  → n_layers × MambaBlock(d_model, d_state, expand)
      [in_proj → depthwise conv1d → SiLU → selective SSM scan → gate → out_proj]
  → Flatten
  → Linear(d_model·seq_len, pred_len·n_sites)
Output (B, pred_len, n_sites)
```

| Hyperparameter | Default |
|---|---|
| `d_model` | 128 |
| `n_layers` | 3 |
| `d_state` | 16 |
| `expand` | 2 |
| `dropout` | 0.1 |

The pure-PyTorch SSM scan runs on CPU or GPU without custom CUDA kernels.
Install `mamba-ssm` to enable the fused CUDA kernel for faster GPU training.

---

## `train.py` options

| Flag | Default | Description |
|---|---|---|
| `--model` | `transformer` | `transformer` or `mamba` |
| `--seq-len` | 24 | Look-back window (hours) |
| `--pred-len` | 6 | Forecast horizon (hours) |
| `--d-model` | 128 | Hidden dimension |
| `--n-layers` | 2 / 3 | Encoder layers |
| `--epochs` | 50 | Max training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--patience` | 8 | Early-stopping patience |
| `--tag` | `""` | Extra string in checkpoint name |

---

## References

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
  https://arxiv.org/abs/1706.03762
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
  https://arxiv.org/abs/2312.00752
- AirNow API — https://www.airnowapi.org/

---

## License

MIT © 2026 Isabelle Lee
