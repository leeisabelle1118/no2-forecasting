# NO2 Forecasting — Transformer & Mamba

Time-series forecasting of ground-level NO₂ concentrations using
**Transformer** and **Mamba (SSM)** models trained on AirNow hourly
observations across ~182–197 US monitoring sites.

---

## Datasets

### AirNow — Ground-level NO₂ observations (model input)

[AirNow](https://www.airnowapi.org/) is a US federal network of **ground-based
air quality monitors** operated by EPA, state, tribal, and local agencies, as
well as Canadian and Mexican partners. Each station continuously measures
pollutant concentrations at ground level and reports hourly averages.

**Why use it for forecasting?** Ground monitors provide dense, continuous
time-series at fixed locations — ideal for training a temporal model. The goal
of this project is to learn the daily and seasonal NO₂ patterns at each site
and forecast the next 1–12 hours from recent history.

| Property | Value |
|---|---|
| Source | [AirNow API](https://www.airnowapi.org/) |
| Data path | `/mnt/data3/AirNow/airnow_no2_YYYYMMDD.nc` |
| Date range | 2023-07-01 → 2024-09-30 (459 days) |
| Sites | ~197 ground monitoring stations |
| Temporal resolution | Hourly (24 observations/day) |
| Variables | `no2` (PPB), `aqi`, `latitude`, `longitude`, `site_name`, `agency` |
| Total timesteps | 10,992 hours |
| Missing data | ~13.7 % overall |
| Spatial bbox | −128→−100 °E, 31→52 °N (Pacific NW to Great Plains) |

NO₂ values are in **parts per billion (PPB)** measured at ground level. The
network spans the Pacific Northwest to the Great Plains and includes sites in
Canada and Mexico. Missing values occur when a monitor is offline for
maintenance or quality-control failures.

**Typical patterns:**
- **Diurnal cycle** — NO₂ peaks in the morning rush hour (~7–9 AM local) due
  to vehicle emissions, dips in the afternoon as sunlight photo-dissociates it,
  and rises again during the evening commute.
- **Seasonal cycle** — Higher in winter (less UV sunlight to break down NO₂,
  more home heating) and lower in summer.
- **Spatial variation** — Urban/industrial sites show much higher levels than
  rural monitors.

**Understanding AQS site codes and notable site types:**

Each monitoring station is identified by an **AQS (Air Quality System) site
code** in the format `SSCCCNNNNN`:
- `SS` — US state FIPS code (e.g. `06` = California, `08` = Colorado)
- `CCC` — county FIPS code
- `NNNNN` — unique site number within the county

Two site types that appear prominently in this dataset:

| Site name | AQS code | Location | What it is |
|---|---|---|---|
| **60 Near Road** | `060710027` | Pomona/Ontario, CA (34.03°N, 117.62°W) | Placed next to **State Route 60** (a major LA-area freeway). "Near Road" is a formal EPA station type — sensors sited directly beside high-traffic roads to measure vehicle-exhaust NO₂, which can be 2–3× higher than background levels just a few hundred metres away. |
| **710 Near Road** | `060374008` | Lynwood, CA (33.86°N, 118.20°W) | Same concept — positioned next to **Interstate 710** (Long Beach freeway), one of the busiest freight corridors in the US. |
| **Anaheim Near Road** | `060590008` | Anaheim, CA (33.82°N, 117.92°W) | Near-road monitor beside a major Anaheim arterial. |
| **Ontario Near Road** | `060710026` | Ontario, CA (34.07°N, 117.53°W) | Near-road monitor in the Inland Empire freight hub. |
| **Portland Near Road** | `410670005` | Portland, OR (45.40°N, 122.75°W) | Near-road monitor on a major Portland corridor. |
| **Globeville** | `080310028` | Denver, CO (39.79°N, 104.99°W) | A neighbourhood in north Denver at the confluence of I-25 and I-70. Historically one of Colorado's most polluted communities — a well-known **environmental justice** site with heavy industrial and highway exposure. |

These sites tend to record the **highest NO₂ values** in the dataset and
therefore appear frequently in "top sites" plots.

---

### TEMPO — Satellite NO₂ column measurements (companion dataset)

[TEMPO](https://tempo.si.edu/) (Tropospheric Emissions: Monitoring of Pollution)
is a NASA geostationary satellite instrument that has been continuously watching
North America from space since 2023. Unlike polar-orbiting satellites that pass
over once per day, TEMPO scans the same area **multiple times per day** (roughly
every 1–2 hours during daylight), making it uniquely suited to track
day-to-day and hour-to-hour pollution events.

> **Note:** TEMPO data is used in the companion
> [tempo-cartopy-visualizations](https://github.com/leeisabelle1118/tempo-cartopy-visualizations)
> project for spatial mapping. It is **not** used as model input here, but
> provides important spatial context for understanding where the AirNow ground
> sites sit relative to satellite-observed NO₂ plumes.

| Property | Value |
|---|---|
| Product | `TEMPO_NO2_L3_V04` (Level-3, gridded) |
| Data path | `/mnt/data3/TEMPO/NO2_L3_V04/` |
| Date range | 2023-08-02 → present |
| Grid | 2950 lat × 7750 lon (~2 km resolution) |
| Coverage | North America (14°→73°N, −168°→−13°E) |
| Temporal resolution | ~hourly scans, daytime only (geostationary) |
| Key variables | `vertical_column_troposphere` (molecules/cm²), `vertical_column_stratosphere`, `main_data_quality_flag` |
| Source | [NASA ASDC DAAC](https://asdc.larc.nasa.gov/project/TEMPO/TEMPO_NO2_L3_V04) |

**How NO₂ is measured differently by each dataset:**

| | AirNow (ground) | TEMPO (satellite) |
|---|---|---|
| What it measures | Surface concentration (PPB) | Total column from surface to space (molecules/cm²) |
| Spatial coverage | 197 point locations | Full North American map |
| Temporal coverage | 24 h/day (continuous) | Daytime only (~7 AM–8 PM local) |
| Spatial resolution | Point observations | ~2 km gridded |
| Best for | Time-series forecasting | Spatial mapping & pollution events |

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
│   ├── 01_explore_airnow.ipynb   # EDA: site map, time series, diurnal cycle,
│   │                             #      missing data, model forward-pass check
│   ├── 02_no2_time_series.ipynb  # Extended time-series plots: top-N sites,
│   │                             #      regional trends, seasonal overlays,
│   │                             #      rolling averages, anomaly detection
│   └── 03_model_diagnostics.ipynb # Diagnostics: error decomposition,
│                                  #      attention weights, residual ACF/PACF
├── outputs/                  # Checkpoints (.pt), training history (.json),
│                             # comparison plots (.png)  — gitignored
├── plots/                    # EDA figures saved by the notebooks — gitignored
├── train.py                  # CLI training script (Transformer or Mamba)
├── predict.py                # Load a checkpoint and forecast a date range
├── compare.py                # Load checkpoints, compare MSE/MAE, save plots
├── environment.yml           # Conda environment spec
├── requirements.txt          # pip fallback
├── GRAPHS_GUIDE.md           # Plain-language guide to every plot (notebooks 01–03)
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

### Inference / prediction

```bash
# Forecast the last week of the test set using the best Transformer checkpoint
python predict.py --model transformer

# Forecast a specific date range and plot named sites
python predict.py --model mamba --start 2024-08-01 --end 2024-08-07 \
    --sites "Seattle-Beacon Hill" "Portland SE Lafayette"

# List available site names
python predict.py --list-sites
```

Saves per-site CSVs and time-series PNGs to `outputs/`.

### Compare models

```bash
python compare.py
```

Generates four files in `outputs/`:

---

## Exploratory Data Analysis (EDA)

Core EDA plots are produced by [`notebooks/01_explore_airnow.ipynb`](notebooks/01_explore_airnow.ipynb).  
Extended time-series visualisations (regional trends, seasonal overlays, rolling averages, anomaly detection) are in [`notebooks/02_no2_time_series.ipynb`](notebooks/02_no2_time_series.ipynb).  
Model diagnostics (error decomposition, attention weights, residual ACF/PACF) are in [`notebooks/03_model_diagnostics.ipynb`](notebooks/03_model_diagnostics.ipynb).  
All figures are saved to `plots/`.

> **Full graph guide:** For a plain-language explanation of every chart across all
> three notebooks — what each graph shows, how to read it, and why it matters for
> NO₂ forecasting — see **[GRAPHS_GUIDE.md](GRAPHS_GUIDE.md)**.

---

### Missing data heatmap

![Missing data heatmap](outputs/eda_missing_data_heatmap.png)

This chart has two panels side by side, both looking at missing NO₂ readings across the monitoring network.

**Left panel — "How much data is each station missing?"**
Each bar shows how many stations are missing a certain percentage of their readings. A tall bar on the left means most stations have good data. A spread-out chart means some stations are much less reliable than others.

**Right panel — "Are data gaps spread evenly over time, or bunched together?"**
The line traces the percentage of stations that had missing readings on each day. Spikes mean many stations went offline at the same time — possibly from a storm, power outage, or scheduled maintenance. A flat line means missingness was steady and random.

---

### Site map

![AirNow monitoring site map](outputs/eda_site_map.png)

A map of the US showing where all ~197 air quality monitoring stations are located. Each dot is one station. The dot color tells you how much NO₂ that station recorded on average — yellow dots measured low NO₂, orange and red dots measured high NO₂. Red dots are usually near cities, highways, or industrial areas. This helps you see which parts of the country have the worst air quality and where the monitoring network is densest.

---

### NO₂ time series — representative sites

![NO₂ time series for representative sites](outputs/eda_no2_time_series.png)

Three line charts stacked on top of each other, each showing the hourly NO₂ readings for one station across the full 15-month dataset (July 2023 – September 2024). The three stations were chosen automatically to represent different extremes:
- **Top** — the station with the highest average NO₂ (likely near a city or highway)
- **Middle** — the station with the lowest average NO₂ (likely rural or coastal)
- **Bottom** — the station with the most complete data (fewest missing readings)

Gaps in the line mean the sensor was offline. Tall spikes mean pollution events. This gives a feel for how much variation exists between individual stations.

---

### Diurnal cycle

![Diurnal NO₂ cycle](outputs/eda_diurnal_cycle.png)

This chart shows the average shape of a typical day's NO₂ levels, averaged across all stations and all days in the dataset. The X-axis is the hour of the day (in UTC), and the Y-axis is NO₂ in parts per billion (PPB). The solid blue line is the average across all stations, and the light blue shaded region shows how much variation there is between stations (10th to 90th percentile). You can see that NO₂ tends to peak in the morning when traffic is heaviest, drops during the afternoon as sunlight breaks it down, then rises slightly again in the evening commute.

---

### Monthly seasonal pattern

![Monthly seasonal NO₂ pattern](plots/monthly_boxplot.png)

A box-and-whisker plot where each box represents one month. Rather than showing a single mean, each box shows the full distribution of monthly-mean NO₂ values **across all monitoring sites**: the centre line is the median, the box spans the interquartile range (IQR), whiskers extend to 1.5×IQR, and dots beyond the whiskers are outlier sites. This makes it easy to see both the seasonal trend and the spread between sites — winter months have higher medians because there is less sunlight to break down NO₂ and more energy burned for heating, while summer shows lower medians but often a wider spread due to wildfire events inflating certain sites.

---

### Daily mean NO₂ time series

**All sites (network-wide):**

![Daily mean NO₂ — all sites](outputs/eda_daily_mean_all_sites.png)

This chart shows the average NO₂ level across all stations for every single day in the dataset. The blue line is the daily average and the shaded region shows ±1 standard deviation — how spread out the stations were on that day. A wide shaded band means stations varied a lot that day; a narrow band means they were all similar. This is useful for spotting unusual events that affected the whole network at once, like a wildfire or heatwave.

**Single site:**

![Daily mean NO₂ — single site](outputs/eda_daily_mean_site.png)

The same type of chart but zoomed in to just one station (`081030006`). Instead of the network average, you see exactly what that one location measured each day. Comparing this to the all-sites chart above lets you see whether a station follows the same trends as the rest of the network or behaves differently due to local conditions.

---

## Training Results (2023-07-01 → 2024-09-30)

24-hour look-back → 6-hour forecast, normalised NO₂, NVIDIA A10G GPU.

| Model | Epochs | Best val MSE | Test MSE | Test MAE | Params |
|---|---|---|---|---|---|
| Transformer | 15 (early stop) | 0.6660 | 1.6057 | 0.6940 | 3.9 M |
| Mamba | 11 (early stop) | 0.7443 | 2.0480 | 0.8766 | 4.0 M |

The **Transformer outperforms Mamba** on this dataset. Mamba's sequential SSM scan is better suited to very long sequences (>512 steps); at 24-hour windows the Transformer's global attention has a natural advantage.

### Loss curves

![Training and validation MSE curves](outputs/comparison_curves.png)

### Predicted vs actual

![Predicted vs actual NO₂ scatter](outputs/comparison_scatter.png)

### Per-site MAE map

![Per-site MAE across monitoring network](outputs/site_mae_map.png)

---

#### `comparison_curves.png` — Training & validation loss curves

Two side-by-side line plots (one per model) showing how MSE changes over training epochs.

- **Training MSE (left panel)** — the loss computed on the training set each epoch.
  A steadily decreasing curve means the model is learning. If it plateaus early,
  try a lower learning rate or more capacity (`--d-model`, `--n-layers`).
- **Validation MSE (right panel)** — the loss computed on held-out data the model
  has never seen during training. This is the most important curve:
  - If it keeps improving alongside training loss → the model is generalising well.
  - If it starts rising while training loss falls → **overfitting**; the model has
    memorised training data. Early stopping (default patience = 8) saves the best
    checkpoint before this happens.
  - A large gap between training and validation MSE also signals overfitting.
- **Learning rate drops** are visible as sudden downward kinks — `ReduceLROnPlateau`
  halves the LR automatically when validation loss stalls for 3 epochs.

---

#### `comparison_scatter.png` — Predicted vs actual NO₂

One scatter plot per model, each showing 5,000 randomly sampled
(actual, predicted) pairs from the test set (values are normalised).

- **Perfect predictions** would lie exactly on the **dashed diagonal** (y = x).
- **Points above the diagonal** → the model over-predicted NO₂.
- **Points below the diagonal** → the model under-predicted NO₂.
- **Tight clustering around the diagonal** = low error.
- **Wide spread or fan shape** = the model struggles at high or low concentrations.
- Compare the two panels: whichever has tighter scatter around the diagonal
  is the better model.

---

#### `site_mae_map.png` — Per-site Mean Absolute Error map

A lat/lon scatter map of all monitoring sites, coloured by their individual
test-set MAE (yellow = low error, red = high error). One map per model.

- **Yellow sites** — the model predicts NO₂ well at those locations.
- **Red/orange sites** — the model struggles; possible reasons include:
  - Highly variable local sources (busy intersections, industrial facilities)
  - More missing data at that site during training
  - The site is at the edge of the network's spatial coverage
- Comparing the two model maps shows **where each architecture fails differently**,
  which can inform ensemble weighting or site-specific fine-tuning.

---

#### `comparison_results.json` — Numeric summary

Machine-readable summary of each model's test performance:
```json
{
  "transformer": { "test_mse": 1.6057, "test_mae": 0.6940, "n_params": 3922590, ... },
  "mamba":       { "test_mse": 2.0480, "test_mae": 0.8766, "n_params": 4020126, ... }
}
```
MSE and MAE are in **normalised units** (each site divided by its training-set
mean), so a MAE of 0.69 means predictions are off by ~69 % of the typical
concentration at that site on average.

---

## Model Diagnostics

Deep-dive diagnostics using the fully-trained checkpoints are in
[`notebooks/03_model_diagnostics.ipynb`](notebooks/03_model_diagnostics.ipynb).
All figures are saved to `plots/`.

### Forecast error time decomposition

![Forecast MAE by hour and month](plots/diag_error_decomp.png)

Two grouped bar charts showing mean absolute error broken down by (left) hour of
day and (right) month of year for both models. Reveals *when* each model makes
its largest errors and whether errors correlate with the diurnal peaks and
seasonal patterns identified in notebooks 01–02.

### Attention weight heatmap

![Transformer attention weights](plots/diag_attention_weights.png)

One heatmap per Transformer encoder layer showing average self-attention weights
over 200 test-set windows. Bright cells at column `t-24` confirm the model has
learned to attend to the same hour from the previous day — the diurnal cycle
identified in notebook 01.

### Residual autocorrelation (ACF / PACF)

![Residual ACF and PACF](plots/diag_residual_acf.png)

2 × 2 grid (ACF and PACF for each model) at lags 0–48 hours. Spikes outside the
95 % confidence band reveal unexplained temporal structure — particularly at lag 24
(daily cycle) and lag 1–6 (short-range persistence).

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
