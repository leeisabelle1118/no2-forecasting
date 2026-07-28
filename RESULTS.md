# NO₂ Forecasting Model Results

**Date:** 2026-07-28  
**Repository:** NO2 Forecasting  
**Dataset:** AirNow NO₂ ground stations (197 sites, 2023-07-01 → 2024-09-30)

---

## 1. Experiment Configuration

### Data Split (Timestamp-Based)

| Partition | Date Range | Duration | Windows |
|---|---|---|---|
| **Training-proper** | 2023-07-01 → 2024-05-31 | 11.2 months | 8,064 |
| **Validation** | 2024-06-01 → 2024-06-30 | 1.0 month | 720 |
| **Test** | 2024-07-01 → 2024-09-30 | 3.1 months | 2,179 |

**Key Design Principles:**
- **Full training window:** 2023-07-01 → 2024-06-30 (12 months, 8,784 windows)
- **Validation is held-out from within the training window** (2024-06-01 → 2024-06-30, 1 month)
- **Test is completely separate** (2024-07-01 → 2024-09-30, never touched during training)
- **Windows assigned by start timestamp** (reproducible, leak-free, stride-independent)
- **Per-site normalization:** Computed exclusively on full 12-month training window (up to 2024-06-30 23:00)

### Model Configuration

| Parameter | Value |
|---|---|
| Sequence length | 24 hours |
| Prediction horizon | 6 hours |
| Stride | 1 hour |
| Number of sites | 197 |
| Device | CPU |

---

## 2. Training Results

### Transformer Model

**Model Details:**
- **Parameters:** 3,922,590
- **Architecture:** Encoder-only attention transformer
- **d_model:** 128, **n_layers:** 2

**Training:**
- **Total epochs:** 15 (with early stopping)
- **Best validation epoch:** Epoch 7 (val_mse = 0.5575)
- **Early stopping:** No improvement after 7 epochs

**Epoch Progression:**
| Epoch | Train Loss | Val MSE | Val MAE | Status |
|---|---|---|---|---|
| 1 | 0.8557 | 0.6315 | 0.4460 | ✓ saved |
| 2 | 0.5801 | 0.5890 | 0.4256 | ✓ saved |
| 3 | 0.5095 | 0.5869 | 0.4365 | ✓ saved |
| 4 | 0.4574 | 0.5730 | 0.4265 | ✓ saved |
| 5 | 0.4247 | 0.5620 | 0.4216 | ✓ saved |
| 6 | 0.3988 | 0.5671 | 0.4242 | — |
| 7 | 0.3758 | 0.5575 | 0.4178 | ✓ **BEST** |
| 8-15 | — | plateaued | — | early stop |

**Test Performance:**
- **Test MSE:** 1.3667
- **Test MAE:** 0.6219
- **Checkpoint:** `transformer_s24_p6_d128.pt`

---

### Mamba Model

**Model Details:**
- **Parameters:** 4,020,126
- **Architecture:** State Space Model (SSM) with selective scanning
- **d_model:** 128, **n_layers:** 2

**Training:**
- **Total epochs:** 11 (with early stopping)
- **Best validation epoch:** Epoch 3 (val_mse = 0.6404)
- **Early stopping:** No improvement after 7 epochs (plateau from epoch 4 onward)

**Epoch Progression:**
| Epoch | Train Loss | Val MSE | Val MAE | Status |
|---|---|---|---|---|
| 1 | 0.7494 | 0.6663 | 0.4754 | ✓ saved |
| 2 | 0.5091 | 0.6411 | 0.4657 | ✓ saved |
| 3 | 0.4329 | 0.6404 | 0.4603 | ✓ **BEST** |
| 4-11 | — | plateaued | — | early stop |

**Test Performance:**
- **Test MSE:** 1.7242
- **Test MAE:** 0.7873
- **Checkpoint:** `mamba_s24_p6_d128.pt`

---

### GNN Model

**Status:** ✅ Completed (latest standalone run)

**Model Details:**
- **Parameters:** 34,054
- **Architecture:** k-NN Graph Convolution + GRU temporal decoder
- **d_model:** 64, **n_layers:** 2, **k_nn:** 5

**Training:**
- **Total epochs:** 50
- **Best validation epoch:** Epoch 47 (val_mse = 0.4792)
- **Checkpoint:** `gnn_s24_p6_d64.pt`

**Test Performance:**
- **Test MSE:** 0.7965
- **Test MAE:** 0.4342

**Implementation note:**
- The adjacency mismatch was fixed by aligning graph construction to `site_codes` from `load_sequences` and keeping self-loops for sites without metadata.

---

## 3. Model Comparison

### Test Set Performance (2024-07-01 → 2024-09-30)

| Model | Parameters | Test MSE | Test MAE | Rank |
|---|---|---|---|---|
| **GNN** | **34,054** | **0.7965** | **0.4342** | 🥇 1st |
| Transformer | 3,922,590 | 1.3667 | 0.6219 | 🥈 2nd |
| Mamba | 4,020,126 | 1.7242 | 0.7873 | 🥉 3rd |

### Performance Gap

| Metric | Best | Second | Gap | Winner |
|---|---|---|---|---|
| **MSE** | 0.7965 (GNN) | 1.3667 (Transformer) | -41.7% | GNN |
| **MAE** | 0.4342 (GNN) | 0.6219 (Transformer) | -30.2% | GNN |

**Key Findings:**
- ✅ **GNN currently has the best test metrics** in the latest run
- ✅ **Transformer remains a strong baseline** and clearly outperforms Mamba
- ✅ **GNN is far smaller** (34k params) than Transformer/Mamba (~4M)
- ✅ **Consistent split boundaries** used across models (12-month train window, 3-month test)

---

## 4. Generated Visualizations

### Comparison Plots (from compare.py)

1. **`comparison_curves.png`**
   - Training and validation MSE curves over epochs
   - Shows Transformer reaching best validation MSE earlier than Mamba
   - Demonstrates early stopping effectiveness

2. **`comparison_scatter.png`**
   - Predicted vs actual NO₂ scatter plots (5,000 sample windows)
   - Transformer shows tighter clustering around diagonal
   - Mamba exhibits wider scatter (higher error variance)

3. **`site_mae_map.png`**
   - Per-site MAE overlaid on geographic map (Albers Equal Area projection)
   - Side-by-side comparison: Transformer vs Mamba
   - Red/orange indicates higher error sites

### GNN-Specific Plots (latest run)

4. **`gnn_training_curves.png`**
   - Training MSE and validation MSE/MAE over 50 epochs
   - Used to confirm convergence stability and absence of severe overfitting

5. **`gnn_scatter.png`**
   - Predicted vs actual NO₂ scatter on test windows
   - Tighter concentration around the diagonal indicates better calibration

### Cartopy Geographic Maps (from cartopy_maps.py)

#### **Baseline (Observations):**
1. **`cartopy_observed_no2.png`**
   - Mean observed NO₂ across all 182+ AIRNOW stations (test period)
   - Viridis colormap showing spatial concentration patterns
   - Identifies pollution hotspots and regional variability

#### **Transformer Results:**
2. **`cartopy_transformer_pred_no2.png`**
   - Mean predicted NO₂ by Transformer model
   - Viridis scale (same as observations for visual comparison)
   - Shows spatial accuracy of learned forecasting patterns

3. **`cartopy_transformer_mae.png`**
   - Per-site Mean Absolute Error (normalized units)
   - Yellow/red colormap highlighting regions of higher error
   - Reveals localized prediction challenges

4. **`cartopy_transformer_bias.png`**
   - Per-site systematic bias (Predicted - Observed, PPB)
   - Diverging RdBu_r colormap (red=overprediction, blue=underprediction)
   - Shows directional prediction tendency by region

#### **Mamba Results:**
5. **`cartopy_mamba_pred_no2.png`**
   - Mean predicted NO₂ by Mamba model
   
6. **`cartopy_mamba_mae.png`**
   - Per-site Mean Absolute Error (normalized units)
   - Higher overall error intensity vs Transformer

7. **`cartopy_mamba_bias.png`**
   - Per-site systematic bias (PPB)
   - More pronounced underprediction in western/central US (blue regions)

### Geographic Insights

- **Observed patterns:** Urban/industrial regions show 2-3x higher NO₂ than rural areas
- **Transformer spatial accuracy:** Captures urban hotspots well with minimal bias
- **Mamba systematic bias:** Tends to underpredict in western regions, compensating elsewhere
- **Model error distribution:** Not uniformly random; clustered near urban centers and coastal areas
- **Regional performance:** Both models perform similarly in rural areas; differences prominent in high-pollution zones

### How to Interpret the Plots

Use this sequence when reading the figures:
1. Start with test metrics (MSE/MAE) to establish overall ranking.
2. Use scatter plots to check calibration against the 1:1 line.
3. Use MAE maps to identify where errors cluster geographically.
4. Use bias maps to detect systematic under/overprediction by region.
5. Use training curves to verify convergence behavior and stability.

---

## 5. Normalization & Denormalization

### Training Mean Computation

```python
# Computed over full 12-month training window (2023-07-01 → 2024-06-30 23:00)
train_mean = np.nanmean(df_train[df_train.index <= FULL_TRAIN_END], axis=0)

# Per-site statistics:
# - Minimum:  0.464 PPB
# - Maximum: 26.099 PPB
# - Mean:     7.605 PPB
# - All 197 sites: valid (no NaN)
```

### Denormalization

```python
# For predictions and actual values:
predicted_ppb = predicted_normalized * train_mean
actual_ppb = actual_normalized * train_mean
```

**Key Feature:**
- ✅ **No data leakage:** Training mean computed only from full 12-month training window
- ✅ **NaN-safe:** Uses `np.nanmean()` to handle ~15% missing data in AirNow dataset
- ✅ **Consistent reference:** Both training and validation normalized by same 12-month window
- ✅ **Validated:** Smoke tests confirm all 197 sites receive valid means

---

## 6. File Structure & Outputs

### Model Checkpoints

```
outputs/
├── transformer_s24_p6_d128.pt              # Trained Transformer weights
├── transformer_s24_p6_d128_history.json    # Training curves, split metadata, test metrics
├── mamba_s24_p6_d128.pt                    # Trained Mamba weights
├── mamba_s24_p6_d128_history.json          # Training curves, split metadata, test metrics
├── gnn_s24_p6_d64.pt                       # Trained GNN weights
└── gnn_s24_p6_d64_history.json             # Training curves, split metadata, test metrics
```

### Comparison & Evaluation Outputs

```
outputs/
├── comparison_results.json                 # MSE, MAE, parameter counts
├── comparison_curves.png                   # Training/validation loss over epochs
├── comparison_scatter.png                  # Predicted vs actual scatter
├── site_mae_map.png                        # Per-site MAE geographic map
├── gnn_training_curves.png                 # GNN train/val curves (latest run)
├── gnn_scatter.png                         # GNN predicted vs actual scatter
└── (geographic visualizations — see below)
```

### Cartopy Geographic Maps

```
outputs/
├── cartopy_observed_no2.png                # Baseline: observed NO₂ across stations
├── cartopy_transformer_pred_no2.png        # Transformer: predicted NO₂
├── cartopy_transformer_mae.png             # Transformer: per-site MAE
├── cartopy_transformer_bias.png            # Transformer: per-site bias (PPB)
├── cartopy_mamba_pred_no2.png              # Mamba: predicted NO₂
├── cartopy_mamba_mae.png                   # Mamba: per-site MAE
└── cartopy_mamba_bias.png                  # Mamba: per-site bias (PPB)
```

---

## 7. Summary & Recommendations

### Key Achievements

✅ **Timestamp-based split successfully implemented** across all three scripts:
   - Training: 12 months (split into 11.2-month proper + 1-month validation)
   - Test: 3 months (completely held-out, never touched during training)
   - Windows assigned by chronological start timestamp (reproducible, leak-free)

✅ **GNN run completed successfully with improved metrics**:
   - Test MSE: 0.7965
   - Test MAE: 0.4342
   - Adjacency mismatch resolved with site-code aligned graph construction

✅ **Transformer remains superior to Mamba**:
   - 19.8% lower MSE than Mamba
   - 21.0% lower MAE than Mamba

✅ **Comprehensive evaluation pipeline**:
   - Automated checkpointing and early stopping
   - Per-site geographic visualization (Cartopy maps)
   - Normalization/denormalization validated

✅ **Geographic insights** via 7 new Cartopy maps:
   - Spatial prediction patterns
   - Per-site error distribution
   - Systematic model biases by region

### Next Steps

1. **Add GNN to automated compare pipeline** so `comparison_results.json` includes all three models.
2. **Hyperparameter Tuning:** tune GNN/Transformer jointly under the same evaluation sweep.
3. **Ensemble Methods:** combine GNN + Transformer predictions.
4. **Uncertainty Quantification:** add prediction intervals (e.g., conformal or Bayesian).
5. **Notebook parity:** keep notebooks 05/06/07 synchronized with script logic.

### File Locations

- **Python scripts:** `/mnt/data3/isybelle1118/NO2 Forecasting/`
- **Checkpoints & outputs:** `/mnt/data3/isybelle1118/NO2 Forecasting/outputs/`
- **Data:** `/mnt/data3/AirNow/`
- **Virtual environment:** `/mnt/data3/isybelle1118/.venv/`

---

## References

### Key Constants (data/load_airnow.py)

```python
TRAIN_END = pd.Timestamp("2024-05-31 23:00")       # Training-proper boundary
FULL_TRAIN_END = pd.Timestamp("2024-06-30 23:00")  # End of 12-month training
```

### Training Commands

```bash
# Transformer
python train.py --model transformer

# Mamba
python train.py --model mamba --seq-len 24 --pred-len 6

# Comparison
python compare.py

# Cartopy maps
python cartopy_maps.py
```

---

**End of Results Report**
