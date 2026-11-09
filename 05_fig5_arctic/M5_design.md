# M5 design document: wind-wave exposure in newly navigable Arctic waters

Compiled: 2026-09-06. Companion code: `M5_pipeline.py` (cells split by `# %%`, can be pasted directly into `M5_analysis.ipynb`).

## 1. Research question and positioning

Existing studies answer either "where can ships go in the Arctic" (POLARIS/ATAM navigability: Wei 2020, Min 2021/2022, Zhao 2024) or "how do Arctic sea states change" (Casas-Prat 2020, Frontiers 2022, Henke 2024). M5 answers: **what are sea states like when ships can go, and are the newly opened waters more dangerous?**

Claimable contributions:
1. Navigation-first conditional statistics: wind-wave metrics are defined on "navigable cell-days", not on open water or the whole domain.
2. "Newly navigable areas" as the unit of analysis.
3. Decomposition of the change in hazardous days: window effect (more navigable days) vs sea-state effect (rougher seas within navigable days).
4. Linking ship class (non-ice-class / PC6 / PC7) to sea state.
5. The information gap in the marginal ice zone (navigable but no wave simulation) reported as a result.
6. Two dataset-level findings: the CSIRO CMIP6 wave ensemble in practice masks waves in the Arctic at about 1% concentration (documentation says 25–75%); artifact grid cells in GCM land channels where WW3 computes waves year-round.

## 2. Data

| Data | Path | Notes |
|---|---|---|
| WW3 wave field hs (3 h, 0.5°) | `<DATA_ROOT>\new_climate\wave\{ACCESS\CDFAC_*, ec_earth\cdfac1\*}` | Meucci et al. 2023/2024 CSIRO ensemble; ACCESS-CM2, EC-Earth3; his 1985–2014, ssp126/585 2015–2100 |
| WW3 wind field u/v (3 h, same grid) | `<DATA_ROOT>\new_climate\criso_wind\{ACCESS-CM2, EC-EARTH3}\{his,126,585}` | Output from the same run as waves; likewise NaN over ice |
| Daily siconc (SIday) | `<DATA_ROOT>\new_climate\siconc\SIday\{ACCESS-CM2, EC-Earth3}\{historical,ssp126,ssp585}` | Downloaded and verified, 224 files, 7.95 GB |
| Daily sithick (SIday) | Same as above, ACCESS-CM2 only | 22 files, 3.77 GB; EC-Earth3 has no future daily thickness |
| Monthly sithick (SImon) | `<DATA_ROOT>\new_climate\siconc\sithick_ECEARTH_*.nc` | EC-Earth3 uses monthly thickness linearly interpolated to daily |
| M2 accident panel results | `M2_results_v2.ipynb` | Hs-bin IRRs used as weights in Section 5 |
| Reference PDFs | `<DATA_ROOT>\references\` | Three Meucci papers + supplements, Müller 2023, Zhao 2024/2025 |

Data not needed: local `sfcWind` daily mean wind (definition mismatch), EC-Earth3-CC wind (not the same member).

## 3. Verified specification facts (for Methods / Supplementary)

- **Target grid**: WW3 0.5° grid, ocean cells with MAPSTA∈{1,7} and ≥60°N, 27 375 cells; shared by both models. MAPSTA in each file is a snapshot at write time (1 active, 7 switched off by ice); only their union is the static ocean mask.
- **WW3 ice threshold**: the supplement states IC0, 25%/75%; a check against daily siconc (1985, 6.7 million cell-days) shows the actual value is about 0.25–0.75%, with κ highest at c≈0.5% (0.88); at c=75% κ is 0.61–0.69 and 37–45% of "<75%" cell-days have no waves. Presumably siconc in percent was not divided by 100. **Waves present ⇔ model siconc≈0 on that day.**
- **Trace ice**: ACCESS-CM2 outputs no sithick where siconc ≤ 0.1%; thickness must be set to 0, otherwise navigable days are underestimated and the contradiction "wave days > navigable days" appears.
- **Artifact grid cells**: about 350 grid cells (1.3%, Canadian Archipelago channels) have waves year-round in WW3 but sea ice frozen year-round, because the GCM ocean grid treats the channels as land; removed uniformly. EC-Earth3 needs a separate diagnosis.
- **EC-Earth monthly thickness interpolation**: verified on ACCESS; navigable cell-days are overestimated by 2% (PC6/PC7) to 4.8% (OW), cell-day agreement 98–99%, per-cell annual-day error median <1 day, 95th percentile <6 days.
- **POLARIS**: original values of IMO MSC.1/Circ.1519 Table 1.3; single ice type, thickness as proxy for ice age (as in Henke 2024, Zhao 2024); RIO ≥ 0 is navigable; no decayed-ice discount, no drift. For non-ice-class ships navigability is in practice driven by concentration (ACCESS thinnest floe ice 0.14 m).
- **Wind-wave definition**: daily maximum (max over 8 steps of 3 h), consistent with M2 (daily maximum of hourly ERA5); 3 h sampling in WW3 is slightly low-biased, i.e. conservative.

## 4. Workflow (the 14 cells of `M5_pipeline.py`)

| Cell | Content | Checks |
|---|---|---|
| 0 | Configuration: paths, periods, sector→passage, bins, POLARIS table, M2 coefficients | Directories exist, file counts correct |
| 1 | Target grid, area weights, passage assignment, sea-ice nearest-neighbour index | 27 375 cells; NN distance median <30 km; FAR point count |
| 2 | Daily siconc/sithick extraction (ACCESS daily thickness; EC-Earth monthly interpolation), uint16 cache | Day counts, trace-ice count, missing share <2% |
| 3 | POLARIS monthly/annual navigable days, longest continuous period | ice-free ⊂ OW ⊂ PC7 ⊂ PC6; monthly/annual consistent |
| 4 | Wave cache: WMO five bins of daily max Hs | Month series aligned with NAV; bins sum to total |
| 5 | Wind cache: six Beaufort bins of daily max U10 + compound wind-wave days (new, 3–4 h per model) | Wind and wave valid days equal per cell per month; bins sum to total |
| 6 | Identify and uniformly remove artifact grid cells `EXCL/VALID` | Sector distribution of artifact cells; remaining "wave > navigable" (monthly scale) <5%, expected 2–5% |
| 7 | Section 1: passage definition and map | Sum of cell counts and areas per passage |
| 8 | Section 2: heatmap of monthly navigable area share (OW/PC7/wave × two scenarios); navigation season length table | 0–100; historical period identical across scenarios |
| 9 | Section 3: wave-field coverage (wave days / OW navigable days) | ≤ 1.02 |
| 10 | Section 4a: newly navigable vs historically navigable; stackplot of the fixed end-of-century new set; Hs/wind speed bin tables and conditional frequencies | Groups mutually exclusive; bins sum to total |
| 11 | Section 4b: decomposition ΔE = ΔN·p̄ + N̄·Δp + ΔN·Δp (Hs≥2.5 m; U10≥17.2 m/s) | Sum of three terms = ΔE |
| 12 | Section 5: M2 hazard-weighted exposure index (monthly, Hs coefficients only, wave days, CI bounds) | ≥1; interval contains point estimate |
| 13 | Export `<DATA_ROOT>\new_climate\_M5\M5_*_{MODEL}.csv` | Files non-empty |

Execution: run with `MODEL='ACCESS'` → change to `'ECEARTH'` and rerun from Cell 0 → show both models side by side, no averaging.

## 5. Definitions

- **passage** (longitude-latitude sectors approximating IHO sea areas, finalised 2026-09-08): R1 NEP = 66.5–82°N, 15°E→180°→156°W (Barents, Kara, Laptev, East Siberian, Chukchi); R2 NWP = 66.5–82°N, 156°W→45°W (Beaufort, Canadian Archipelago channels, Baffin Bay); R3 TSR = ≥82°N; R4 Fram = 66.5–82°N, 45°W→15°E (Greenland Sea). NEP/NWP boundary at Point Barrow. Area-weighted, cell area ≈ 3090·cos(lat) km².
- **Navigable day**: POLARIS RIO ≥ 0 (main: non-ice-class ships; secondary: PC7, PC6); sensitivity: SIC < 15%.
- **Newly navigable grid cell**: mean annual OW navigable days ≥ 90 in the future period and < 30 in 1985–2014; historically navigable: ≥ 90 days in 1985–2014. Time series use the fixed set of cells that "become newly navigable by 2081–2100".
- **Navigable wave day**: a navigable day with non-NaN hs, a subset of navigable days (the ice-free part). Wind-wave metrics are defined only on these days; coverage must always be reported.
- **Decomposition**: N = wave days, E = wave days above threshold, p = E/N.
- **Weighted index**: I = Σ nvalid·exp(Σ_k β_k n_k) / Σ nvalid, β_k = ln(IRR_k), M2 Table 1 Hs bins (A_it specification), monthly scale.

## 6. Known results (ACCESS-CM2, after corrections)

- September navigable area for non-ice-class ships reaches 98–99% in 2041–2060, with no difference between scenarios; scenario differences are concentrated in winter-spring and after 2060; SSP1-2.6 plateaus after 2061.
- Navigation season expansion is asymmetric: November–December opens faster than March–May; May is the bottleneck.
- The value of ice class is concentrated in winter-spring; in summer there is no difference from the 2040s onward.
- Coverage: wave days / OW navigable days, historical 0.74; SSP1-2.6 July–August end of century 0.57–0.63; SSP5-8.5 end of century August–December ≥0.95.
- Share of wave days with Hs ≥ 2.5 m in newly navigable areas: SSP1-2.6 stays at about 12% (pure window effect); SSP5-8.5 rises from 11% to 17.5% (sea-state effect added), close to today's Barents Sea (about 16%).

## 7. Limitations and to-do

- Only two GCMs with one member each; ACCESS-CM2 melts ice fast (ECS≈4.7 K).
- Insufficient resolution in the NWP; the wave field is unusable in the archipelago; for the NWP only navigability is reported.
- No drift, icebreakers or decayed-ice discount; EC-Earth thickness is monthly-interpolated.
- No bias correction of the wave field; suggest using ERA5 2002–2014 over historical open water to assess differences in WMO bin frequencies.
- M2 parameters are transferred only for weighting and threshold choice, not for accident prediction; consider adding a Nordic–Barents region in M2 as a high-latitude parameter anchor (about 1 170 accidents north of 60°N, about 700 in 2011–2021).
- Optional supplement for the marginal ice zone wave field: Henke 2024 pan-Arctic WW3 (EC-Earth3 SSP5-8.5 2020–2069, ice threshold 50%, arcticdata.io).
- To-do: ACCESS wind cache; full EC-Earth workflow and artifact-cell diagnosis; notebook comparing both models; corridor bottleneck analysis (optional).
