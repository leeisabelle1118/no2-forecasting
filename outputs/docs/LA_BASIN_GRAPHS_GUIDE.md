# LA Basin NO₂ Analysis — Graphs Guide

All figures in this directory are produced by two sources:
- **`la_basin_no2_analysis.ipynb`** — EPA AQS daily data, 2023–2024
- **`notebooks/04_graphmamba_vs_observed.ipynb`** — GraphMamba predicted vs AirNow observed, Jul–Sep 2024

---

## EPA AQS Analysis (`la_basin_no2_analysis.ipynb`)

### `map_site_classification.png`
**What it shows:** All EPA AQS monitoring stations in the LA Basin, coloured by whether the site has complete meteorological instrumentation (wind, temperature, pressure, RH, dew point) alongside the NO₂ sensor.

**Layers:** Urban footprints · rivers · county lines · major roads · city labels.

**Why it matters:** Sites with full met coverage can support feature-rich models. Sites with NO₂ only can still be included as targets, but meteorological covariates must come from neighbour interpolation.

---

### `map_max_no2_2024.png`
**What it shows:** Each monitoring station in the LA Basin coloured by its maximum observed daily mean NO₂ in 2024. Higher values (orange/red) mark chronic hotspots.

**Pattern to note:** Near-road sites (Ontario, 710 freeway) consistently show the highest maxima, confirming that traffic proximity is the primary driver of extreme events in the basin.

---

### `map_focus_sites.png`
**What it shows:** The three focus sites selected for detailed analysis — **Anaheim**, **Ontario Near Road (Etiwanda)**, and **Simi Valley-Cochran Street** — overlaid on the full site network.

**Why these three:** They span the basin's dominant gradients: coastal/urban (Anaheim), near-highway industrial (Ontario), and upwind suburban (Simi Valley), making them representative for a basin-level comparison.

---

### `ts_anaheim.png` · `ts_ontario.png` · `ts_simi.png`
**What they show:** Daily mean NO₂ (ppb) over January 2023 – January 2025 for each focus site. The horizontal dashed red and dotted green lines mark the time-series mean and median.

The panel below each time series is a Cartopy inset map (urban areas · county lines · roads) locating the site within the basin.

**Summary stats box:** Mean, median, min–max range, and number of valid days are printed in the top-left corner.

**Pattern to note:** Ontario shows the highest amplitude and the clearest summer trough / winter peak seasonal cycle. Simi Valley shows the lowest absolute concentrations but a proportionally similar seasonal shape.

---

### `hist_anaheim.png` · `hist_ontario.png` · `hist_simi.png`
**What they show:** Frequency distribution of daily mean NO₂ values for each focus site. Bin width = 0.5 ppb.

The red and green dashed lines mark mean and median. The stats box (upper right) also includes mode, standard deviation, min, and max.

**Pattern to note:** Ontario's distribution is strongly right-skewed with a heavy tail, confirming occasional extreme high-NO₂ events. Anaheim and Simi Valley are more symmetric.

---

## GraphMamba Model Evaluation (`notebooks/04_graphmamba_vs_observed.ipynb`)

> **Data:** `/mnt/data3/GraphMamba/output/test/predictions.nc`  
> **Time period:** 2024-07-01 → 2024-09-30 (hourly TEMPO-paced inference)  
> **Sites:** 21 South Coast AQMD stations within the LA Basin bounding box  
> **Note:** `observed_no2` in the NC file comes from the AirNow dataset (different from EPA AQS used above).

---

### `gm04_la_sites_map.png`
**What it shows:** Cartopy map of all 21 LA Basin sites present in the GraphMamba inference output, coloured by their mean AirNow observed NO₂ over the inference period.

**Use:** Quick spatial sanity check — confirms the inference covered both coastal low-NO₂ sites and the near-road high-NO₂ sites in the San Bernardino/Riverside corridor.

---

### `gm04_ts_anaheim.png` · `gm04_ts_ontario.png`
**What they show:** Hourly time series of GraphMamba predicted NO₂ (red dashed line) vs AirNow observed NO₂ (coloured scatter points) for Anaheim and Ontario Near Road.

The inset Cartopy map below each plot shows the site's location within the basin.

The metrics box (upper left) shows hourly RMSE, MAE, R², and number of valid observation–prediction pairs.

**What to look for:**
- Does the model capture the diurnal amplitude?
- Are spikes from high-traffic events captured or missed?
- Does bias differ between Anaheim (lower baseline) and Ontario (higher baseline)?

---

### `gm04_daily_anaheim.png` · `gm04_daily_ontario.png`
**What they show:** Daily mean NO₂ (aggregated from hourly data, minimum 4 valid obs/day required) for predicted vs observed. This makes the signal easier to interpret and directly comparable to the AQS daily data in `ts_*.png`.

**Why aggregate:** Hourly TEMPO-paced predictions have irregular time steps (some hours have gaps); daily means smooth this and highlight multi-day pollution episodes.

---

### `gm04_scatter_all_sites.png`
**What it shows:** Hexbin scatter of all valid (obs, pred) hourly pairs pooled across all 21 LA Basin sites. The 1:1 line is overlaid. Colour intensity = number of points in each hexagon.

**Interpretation guide:**
- Points above the 1:1 line → model over-predicts
- Points below → model under-predicts
- Spread at high observed values → model struggles with extreme events

The overall RMSE, MAE, R², and total sample size are printed in the annotation box.

---

### `gm04_la_rmse_map.png`
**What it shows:** Each LA Basin site coloured by its per-site hourly RMSE (green = low error, red = high error) on the same Cartopy base map.

**Pattern to note:** Near-road and industrial sites tend to have higher RMSE because traffic spikes are harder to predict from spatially averaged TEMPO columns. Coastal and suburban sites typically have lower RMSE.

---

### `gm04_la_site_metrics.csv`
Per-site table with columns: Site, Lat, Lon, RMSE (PPB), MAE (PPB), R², N (valid obs). Sorted by RMSE ascending.

---

## Comparison: AQS vs AirNow at Anaheim & Ontario

| Metric | AQS daily (2023–2024) | AirNow hourly (Jul–Sep 2024) |
|--------|----------------------|------------------------------|
| Time coverage | 2 years | 3 months |
| Temporal resolution | Daily | Hourly |
| Use in this repo | EPA AQS analysis | GraphMamba training/evaluation |
| Anaheim mean | see `ts_anaheim.png` stats | see `gm04_ts_anaheim.png` stats |
| Ontario mean | see `ts_ontario.png` stats | see `gm04_ts_ontario.png` stats |

Both datasets show Ontario Near Road with consistently higher NO₂ than Anaheim, confirming spatial consistency between the two measurement networks.
