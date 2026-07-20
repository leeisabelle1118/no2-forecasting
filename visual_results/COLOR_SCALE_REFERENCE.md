# Model-Specific Color Scale Reference

## Implementation Summary

All Cartopy geographic maps now use **model-optimized color scales** computed independently for each model. This standardization ensures consistent colors **within each model's suite of three maps** (prediction, error, bias) while allowing each model to show its own data range clearly.

**Key Insight:** Mamba's color scales are different from Transformer's because the models produce different ranges of predictions, errors, and biases. This is intentional—it lets you see each model's performance in its own context.

---

## Color Scale Details

### 1. TRANSFORMER Suite — NO₂ Predictions (Viridis)

**Model-specific range:** 0.00 – 23.75 PPB  
**Applied to:**
- `cartopy_transformer_pred_no2.png`

**Color Interpretation:**
| Color | NO₂ Level | Environment |
|-------|-----------|-------------|
| 🟢 Green | 0–5 PPB | Clean air, rural areas |
| 🟡 Yellow | 5–12 PPB | Moderate pollution, semi-urban |
| 🟠 Orange | 12–24 PPB | High pollution, urban areas |

---

### 2. MAMBA Suite — NO₂ Predictions (Viridis)

**Model-specific range:** 0.00 – 29.97 PPB  
**Applied to:**
- `cartopy_mamba_pred_no2.png`

**Color Interpretation:**
| Color | NO₂ Level | Environment |
|-------|-----------|-------------|
| 🟢 Green | 0–6 PPB | Clean air, rural areas |
| 🟡 Yellow | 6–15 PPB | Moderate pollution, semi-urban |
| 🟠 Orange | 15–30 PPB | High pollution, urban areas |

**Note:** Mamba's range extends higher (30 PPB) than Transformer's (24 PPB) because Mamba tends to overpredict in high-pollution areas.

---

### 3. OBSERVED — NO₂ Baseline (Viridis)

**Observed range:** 0.00 – 18.61 PPB  
**Applied to:**
- `cartopy_observed_no2.png`

**Color Interpretation:**
| Color | NO₂ Level | Environment |
|-------|-----------|-------------|
| 🟢 Green | 0–4 PPB | Clean air, rural areas |
| 🟡 Yellow | 4–9 PPB | Moderate pollution, semi-urban |
| 🟠 Orange | 9–19 PPB | High pollution, urban areas |

---

### 4. TRANSFORMER Suite — Error Maps (Yellow-Orange-Red)

**Model-specific range:** 0 – 1.383 (normalized units)  
**Applied to:**
- `cartopy_transformer_mae.png`

**Color Interpretation:**
| Color | MAE | Quality |
|-------|-----|---------|
| 🟡 Yellow | 0–0.46 | Excellent (low error) |
| 🟠 Orange | 0.46–0.92 | Good (moderate error) |
| 🔴 Red | 0.92–1.38 | Challenging (higher error) |

---

### 5. MAMBA Suite — Error Maps (Yellow-Orange-Red)

**Model-specific range:** 0 – 1.718 (normalized units)  
**Applied to:**
- `cartopy_mamba_mae.png`

**Color Interpretation:**
| Color | MAE | Quality |
|-------|-----|---------|
| 🟡 Yellow | 0–0.57 | Excellent (low error) |
| 🟠 Orange | 0.57–1.15 | Good (moderate error) |
| 🔴 Red | 1.15–1.72 | Challenging (higher error) |

**Note:** Mamba's error scale is wider (1.72 vs 1.38) because it makes larger prediction errors overall.

---

### 6. TRANSFORMER Suite — Bias Maps (Red-Blue Diverging)

**Model-specific range:** −9.37 to +9.37 PPB  
**Centered at:** 0 PPB (white)  
**Applied to:**
- `cartopy_transformer_bias.png`

**Color Interpretation:**
| Color | Bias | Meaning |
|-------|------|---------|
| 🔴 Red | +1 to +9.37 PPB | Overprediction (model too high) |
| ⚪ White | ≈ 0 PPB | Unbiased (accurate) |
| 🔵 Blue | −1 to −9.37 PPB | Underprediction (model too low) |

---

### 7. MAMBA Suite — Bias Maps (Red-Blue Diverging)

**Model-specific range:** −15.30 to +15.30 PPB  
**Centered at:** 0 PPB (white)  
**Applied to:**
- `cartopy_mamba_bias.png`

**Color Interpretation:**
| Color | Bias | Meaning |
|-------|------|---------|
| 🔴 Red | +1 to +15.30 PPB | Overprediction (model too high) |
| ⚪ White | ≈ 0 PPB | Unbiased (accurate) |
| 🔵 Blue | −1 to −15.30 PPB | Underprediction (model too low) |

**Note:** Mamba's bias scale is wider (±15.30 vs ±9.37) because it has systematic underprediction tendencies in some regions.

---

## Why Model-Specific Scales?

**Benefit:** Each model's maps form a cohesive visual suite where colors reflect that model's performance and data distribution.

| Aspect | Benefit |
|--------|---------|
| **NO₂ Prediction Maps** | Same yellow on Transformer = 12 PPB; same yellow on Mamba = 15 PPB. Shows regional pollution in each model's context. |
| **Error Maps** | Easier to see which model has more yellow (fewer errors) in specific regions without color compression. |
| **Bias Maps** | Red/blue dominance is clear for each model independently. Transformer mostly white = well-calibrated; Mamba more blue = systematic underprediction. |

**Comparison Approach:** Look at the same region across models:
- **Transformer orange, Mamba orange** = Both see high pollution (aligned predictions)
- **Transformer orange, Mamba yellow** = Transformer sees higher pollution (Transformer overpredicts or more accurate)
- **Transformer yellow, Mamba orange** = Mamba sees higher pollution (Mamba overpredicts)

---

## Scale Calculation Method

**Per-Model Scales:**
- Compute 95th percentile of each model's predictions separately
- Compute 95th percentile of each model's MAE separately
- Compute symmetric ±95th percentile of each model's absolute bias separately

**Observed Scale:**
- Compute 95th percentile of observed NO₂ only

---

## Technical Details

### Cartopy Projection
- **Type:** Albers Equal Area
- **Center:** 96°W, 37.5°N (central USA)
- **Standard Parallels:** 29.5°N, 45.5°N
- **Extent:** 125°W – 66°W, 24°N – 50°N (continental US)

### Features
- State boundaries (gray lines)
- Coastlines (dark lines)
- Lakes (light blue)
- Ocean (light blue background)
- Monitoring stations (colored dots overlaid on map)

### Resolution
- **DPI:** 150 (print-quality)
- **Format:** PNG
- **Typical file size:** 230–240 KB per map

---

## Consistency Checklist

✅ Transformer's 3 maps (pred, MAE, bias) use consistent Transformer scales  
✅ Mamba's 3 maps (pred, MAE, bias) use consistent Mamba scales  
✅ Observed map uses its own independent scale  
✅ All colorbars display explicit ranges  
✅ All maps use same geographic projection and extent  
✅ All maps share consistent styling (states, coastlines, water features)  
✅ Model-specific scales allow independent performance context viewing

---

**Generated:** 2026-07-20 (Updated for model-specific scales)  
**Python Script:** `cartopy_maps.py`  
**Output Directory:** `outputs/` (and copied to `visual_results/`)  
**Color Scale Philosophy:** Per-model optimization for clarity within each model's suite

---

## How to Use These Maps

### **Comparing Predictions to Observations**
1. Look at `cartopy_observed_no2.png` (baseline)
2. Compare to `cartopy_transformer_pred_no2.png` and `cartopy_mamba_pred_no2.png`
3. **Same colors = accurate predictions** (both models and observation align)
4. **Transformer orange where observed is yellow** = Transformer overpredicts
5. **Mamba yellow where observed is orange** = Mamba underpredicts

### **Comparing Model Performance**
1. Look at `cartopy_transformer_mae.png` and `cartopy_mamba_mae.png`
2. **Yellow > Orange > Red** = Look for which model has more yellow (fewer errors)
3. **Red hotspots** = Both models struggle; likely urban complexity or missing meteorology

### **Identifying Systematic Bias**
1. Look at `cartopy_transformer_bias.png` and `cartopy_mamba_bias.png`
2. **Transformer mostly white** = Well-calibrated, unbiased
3. **Mamba mostly blue** = Systematically underpredicts
4. **Red in west, blue in east** = Geographic bias pattern

---

## Example Interpretations

### Regional Comparison: LA Basin (High Pollution)

**Observed:** Bright orange (~20–24 PPB)

| Model | Predicted Map | MAE Map | Bias Map | Verdict |
|-------|---|---|---|---|
| **Transformer** | Orange (bright) | Yellow | Slight red | ✅ Accurate |
| **Mamba** | Yellow-orange (muted) | Orange | Blue | ⚠️ Underpredicts |

**Conclusion:** Transformer captures LA basin pollution intensity; Mamba underestimates.

---

### Regional Comparison: Rural Texas (Low Pollution)

**Observed:** Green (~2–5 PPB)

| Model | Predicted Map | MAE Map | Bias Map | Verdict |
|-------|---|---|---|---|
| **Transformer** | Green | Yellow | White | ✅ Accurate |
| **Mamba** | Green | Yellow | White | ✅ Accurate |

**Conclusion:** Both models perform equally well in low-pollution zones.

---

## Scale Calculation Method

**NO₂ Scale:**
- Denormalize observations and all model predictions to PPB
- Compute 95th percentile across all values
- Use 0 (minimum) to 95th percentile for vmin/vmax

**Error Scale:**
- Compute MAE for all model predictions
- Use 0 to 95th percentile of MAE values

**Bias Scale:**
- Compute bias (predicted - observed) in PPB for all models
- Use symmetric scale: ±95th percentile of absolute bias
- Centered at zero for diverging colormap

---

## Technical Details

### Cartopy Projection
- **Type:** Albers Equal Area
- **Center:** 96°W, 37.5°N (central USA)
- **Standard Parallels:** 29.5°N, 45.5°N
- **Extent:** 125°W – 66°W, 24°N – 50°N (continental US)

### Features
- State boundaries (gray lines)
- Coastlines (dark lines)
- Lakes (light blue)
- Ocean (light blue background)
- Monitoring stations (colored dots overlaid on map)

### Resolution
- **DPI:** 150 (print-quality)
- **Format:** PNG
- **Typical file size:** 230–240 KB per map

---

## Consistency Checklist

✅ All NO₂ maps (observed + 2 models) use same viridis scale (0–25.37 PPB)  
✅ All MAE maps (2 models) use same YlOrRd scale (0–1.661 norm units)  
✅ All bias maps (2 models) use same diverging scale (±13.71 PPB)  
✅ All colorbars display explicit range  
✅ All maps use same geographic projection and extent  
✅ All maps share consistent styling (states, coastlines, water features)

---

**Generated:** 2026-07-20  
**Python Script:** `cartopy_maps.py`  
**Output Directory:** `outputs/` (and copied to `visual_results/`)
