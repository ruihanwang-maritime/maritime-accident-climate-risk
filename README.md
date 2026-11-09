# Shipping accident risk under severe marine weather in a changing climate — analysis code

Code accompanying the manuscript *"Shipping accident risk under severe marine weather and climate change"*
(Wang, Verschuur, Zhang & Kim). Folders follow the paper: main-text Figs 1–5, Methods Eqs 1–11, and the
Supplementary Information.

## Data availability

**This repository contains code only. No data, model outputs or accident records are included.**

| Source | Used for | Access |
|---|---|---|
| Lloyd's List Intelligence (LLI) Casualty Statistics | accident counts, outcomes, vessel attributes (Figs 1–3) | Licensed; cannot be redistributed. Access must be arranged with LLI |
| Global Maritime Traffic Density Service (GMTDS), monthly cargo-vessel AIS density | shipping activity `E_it` (Figs 2, 4) | From the provider |
| World Bank Global Shipping Traffic Density | Fig. 1 | Open |
| ERA5 hourly significant wave height and 10-m wind | observed weather (Figs 2–4) | Open, Copernicus Climate Data Store |
| CSIRO CMIP6 WAVEWATCH III wave and wind fields (ACCESS-CM2, EC-Earth3) | projections (Figs 4–5) | Open, CSIRO Data Access Portal |
| CMIP6 sea-ice concentration and thickness (ACCESS-CM2, EC-Earth3) | Arctic navigability (Fig. 5) | Open, Earth System Grid Federation |

Notebooks are published **without outputs**.

### Data paths

Every script and notebook resolves its inputs relative to one data folder, read from the environment
variable `WRH_DATA_ROOT` (the default `G:/WRH_data` is the authors' machine). Set it before launching Python or Jupyter:

```bash
export WRH_DATA_ROOT=/path/to/data          # Linux / macOS
$env:WRH_DATA_ROOT = "D:\path\to\data"      # Windows PowerShell
```

Expected sub-folders under `WRH_DATA_ROOT`: `AIS_processing/accident_data` (LLI-derived, restricted),
`AIS_processing/weather_data/{wave,wind}` (ERA5), `climate_AIS_data/GMTDS` (GMTDS rasters),
`AIS_processing/panel_cache_M2` (panel caches written by `M2_merge.ipynb`), and `new_climate/` (CMIP6/WW3 inputs and
projection outputs). Fig. 3 outputs are written next to `M3_reanalysis_complete.py` (`M3_reanalysis_outputs/`);
set `M3_OUT_DIR` if they live elsewhere.

## Environment

Python 3.10. Create the environment with `conda env create -f environment.yml`
(main packages: pyfixest 0.60, statsmodels, pandas, pyarrow, xarray, netCDF4, rasterio, cartopy, matplotlib, scipy).

## Layout

```
00_data_preparation/              raw sources -> analysis-ready inputs
01_fig1_global_patterns/          Fig. 1, Methods "Spatial concentration and accident–activity scaling" (Eq 1)
02_fig2_weather_and_occurrence/   Fig. 2, Methods "Exposure-adjusted associations with marine weather" (Eqs 2–3)
03_fig3_accident_consequences/    Fig. 3, Methods "Accident consequence models" (Eq 4)
04_fig4_projected_occurrence/     Fig. 4, Methods "Projected changes across major shipping regions" (Eqs 5–6)
05_fig5_arctic/                   Fig. 5, Methods "Arctic navigability and potential weather exposure" (Eqs 7–11)
06_supplementary/                 SI figure script and an index of where each SI item is computed
```

Weather categories are shared across chapters: significant wave height (WMO sea-state classes)
<1.25, 1.25–2.5, 2.5–4, 4–6, 6–9, ≥9 m; 10-m wind speed (Beaufort) <8, 8–10.8, 10.8–13.9, 13.9–17.2, 17.2–20.8, ≥20.8 m s⁻¹.

---

## 00_data_preparation

### 01_accident_records (requires the licensed LLI extract)

| Order | File | Output |
|---|---|---|
| 0 | `build_accident_parsed.py` | All LLI records with decimal coordinates and date fields (Fig. 1 input) |
| 1 | `build_full_accident.py` | All vessel types 2011–2021, DMS → decimal coordinates |
| 2 | `clean_cargo_accident.py` | Cargo vessels, tankers excluded, 2002–2021 (input to Fig. 2) |
| 3 | `match_weather_accident.py` | Each 2011–2021 accident matched to ERA5 daily maximum Hs and U10 (nearest point; ≤1° search for waves) |
| 4 | `add_vessel_age.py` | Vessel age, class and manager joined back from the LLI extract by original row index (input to Fig. 3) |

### 02_shipping_activity_GMTDS

| Order | File | What it does |
|---|---|---|
| 1 | `process_0719_data.py`, `prepare_0725_data.py`, `add_072555_to_regions.py` | Unzip, fix CRS and rename downloaded batches |
| 2 | `organize_gmtds.py`, `bucket_to_6region.py`, `reorganize_by_target.py` | Classify tiles by their true bounds, deduplicate, mosaic and crop to the six study regions |
| 3 | `organize_cargo_ship.py` | Cargo-only monthly rasters + manifest (read by `M2_merge.ipynb`) |
| — | `visualize_gmtds_coverage.py`, `check_GMTDS.ipynb` | Coverage maps and month counts |
| — | `regional_bounding.ipynb` | Six regions and their subregions (Fig. 2a; Supplementary Table 2) |

### 03_ERA5_reanalysis

`weather_data_wave_download.ipynb`, `weather_data_wind_download.ipynb` (hourly Hs, u10, v10, 2002–2021) and
`wave_era5_his_download.ipynb`, `wind_era5_his_download.ipynb` (earlier years).

### 04_climate_projections_and_bias_correction

| File | What it does |
|---|---|
| `wind_criso_download.ipynb` | CSIRO WAVEWATCH III wind fields (ACCESS-CM2, EC-Earth3; historical / SSP1–2.6 / SSP5–8.5) |
| `sic_sim_download.ipynb` | CMIP6 sea-ice concentration and thickness for the Arctic analysis |
| `wind_wave_QDM_test.ipynb` | Regional quantile delta mapping (QDM) on the study regions, 2° cells, calibration 2002–2014 (Fig. 4b–c) |
| `global_qdm.py` | The same QDM on all 2° ocean cells (Fig. 4a; Supplementary Fig. 6); `python global_qdm.py --stage all` |
| `wave_bias_correction.ipynb` | Early wave-only QDM test |

---

## 01_fig1_global_patterns — `M1_results_v3.ipynb`

Global shipping density vs. all accidents 2002–2021 on a 1° grid: maps (a–b), cumulative concentration curves (c),
PPML scaling with HC1 standard errors (d, Eq 1). Analysis domain: ocean cells with ≥ 87,600 annualized AIS position
reports (`FLOOR`; Supplementary Section 1.2).

## 02_fig2_weather_and_occurrence

| Order | File | What it does |
|---|---|---|
| 1 | `M2_merge.ipynb` | 2° grid-cell × month panel: accident counts by type, GMTDS cargo activity `E_it`, ERA5 daily cell-maximum Hs/U10, category-day counts. Cell 2 clears the cache |
| 2 | `M2_update_v3.ipynb` | Category models (Eq 2, Fig. 2b left), threshold models (Fig. 2b right), compound models (Eq 3, Fig. 2c), accident-type tiles with 95% CI (Fig. 2d) |
| SI | `M2_merge_grid1.py`, `M2_grid1_vs_grid2.py` | 1° panel and 1° vs 2° comparison (Supplementary Fig. 2) |

PPML (`pyfixest.fepois`); fixed effects grid cell × calendar month + region × year-month; exposure offset
`log(E_it + 0.5)`; standard errors clustered by grid cell.

## 03_fig3_accident_consequences

| File | What it does |
|---|---|
| `M3_reanalysis_complete.py` | Linear probability models for serious outcome and total loss on cargo-vessel accidents (Eq 4); region × calendar-month + year FE, SE clustered on 5° cells; age and size effect modification; joint age-and-size interaction models |
| `M3_v4.ipynb` | Fig. 3 panels; sample audit and Supplementary Table 6 (cell 0) |

## 04_fig4_projected_occurrence

| Order | File | Output |
|---|---|---|
| 1 | `wind_wave_pro_variation.ipynb` | Region × calendar-month category-day counts per period and rate ratios with the Fig. 2 coefficients (Eq 5) |
| 2 | `vis_pro_extreme_wind_wave.ipynb` | Annual threshold-exceedance series and seasonal aggregates |
| 3 | `M4_test.ipynb` | Series re-based to each model–scenario's own 2021–2040; category-bin series |
| 4 | `M4_v1.ipynb` (cell 0) | Regional and seasonal tables re-based to 2021–2040 with fixed activity weights (Eq 6) |
| 5 | `make_map_days.py` | Fig. 4a input: per-cell change in Hs ≥ 6 m / U10 ≥ 17.2 m s⁻¹ days, with artefact mask |
| 6 | `M4_v2.ipynb` | Final panels: a maps, b threshold series, c seasonal heat maps |
| alt | `annual_exceedance_curve.py`, `season_heat.py` | Script versions of panels b and c |
| text | `fig4c_text_counts.py` | Counts quoted in the Results for Fig. 4c |

## 05_fig5_arctic

| File | What it does |
|---|---|
| `M5_design.md` | Design notes: data, data-quality checks (WW3 ice masking, spurious channel cells), definitions |
| `M5_pipeline.py` | Cells 0–13: target grid, daily sea ice, POLARIS navigability (Eq 7), wave/wind category caches, exposure (Eqs 8–9), decomposition, occurrence index inputs |
| `M5_data.ipynb` | Notebook version of the pipeline (run once per model: `MODEL='ACCESS'`, then `'ECEARTH'`) |
| `M5_analysis.ipynb` | Checks on the cached outputs |
| `M5_vis_v2.ipynb` | Fig. 5: a first-open decade maps, b area-integrated exposure, c occurrence index (Eqs 10–11) |
| `check_sithick_interpolation.py` | Daily vs. interpolated monthly sea-ice thickness check (Supplementary Table 8) |

The Fig. 5c index uses all five non-reference category coefficients from the Fig. 2 models
(`IRR_HS`, `IRR_U` in `M5_vis_v2.ipynb`; Supplementary Table 10).

## 06_supplementary

`si_figures.py` draws Supplementary Figs. 1–14 from the cached analysis outputs (functions `fig_S1_…` to `fig_S14_…`);
`M2_estimation_samples.py` produces Supplementary Table 5. `06_supplementary/README.md` maps each SI section, table and
figure to the code that computes it.

## Licence

MIT (see `LICENSE`). If you use this code, please cite the article (see `CITATION.cff`).
