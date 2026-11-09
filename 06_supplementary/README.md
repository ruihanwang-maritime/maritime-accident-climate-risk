# Supplementary Information — where each item is computed

Numbering follows the final Supplementary Information. SI figures are drawn by `si_figures.py`
(`fig_S<n>_…` functions; run the file, or import it and call one function). Functions named `fig_X<n>_…` produce drafts that
are not part of the final Supplementary Information.

| SI item | Code location |
|---|---|
| Section 1.1, Supplementary Table 1 — analytical samples | `01_fig1_global_patterns/M1_results_v3.ipynb` (global sample); `02_fig2_weather_and_occurrence/M2_merge.ipynb` (occurrence panel); `03_fig3_accident_consequences/M3_v4.ipynb` cell 0 (consequence sample audit) |
| Section 1.2 — global activity reference and analysis domain (≥ 87,600 AIS reports per year) | `M1_results_v3.ipynb` (`YEARS = 74/12`, `FLOOR`) |
| Section 1.3 — monthly activity and occurrence panel | `M2_merge.ipynb` (`E_it`, `log_E = log(E_it + 0.5)`); `00_data_preparation/02_shipping_activity_GMTDS/` |
| Section 1.4 — cargo-vessel selection and consequence definitions | `00_data_preparation/01_accident_records/clean_cargo_accident.py`; `M3_v4.ipynb` cell 0 (`build_analysis_sample`) |
| Section 1.5, Supplementary Table 2, Supplementary Fig. 1 — study regions | `00_data_preparation/02_shipping_activity_GMTDS/regional_bounding.ipynb`; `si_figures.py` `fig_S1_region_maps` |
| Section 2.1 — weather matching | `M2_merge.ipynb` cells 8–11 (occurrence); `00_data_preparation/01_accident_records/match_weather_accident.py` (consequence) |
| Section 2.2, Supplementary Tables 3–4 — categories and thresholds | `02_fig2_weather_and_occurrence/M2_update_v3.ipynb` (`WAVE_CUTS`, `WIND_CUTS`, `ABS_GRID`) |
| Section 3.1, Supplementary Table 5 — estimation samples | `M2_estimation_samples.py` |
| Section 3.2, Supplementary Fig. 2 — 1° × 1° sensitivity | `02_fig2_weather_and_occurrence/M2_merge_grid1.py` → `M2_grid1_vs_grid2.py`; `fig_S2_resolution` |
| Section 4.1, Supplementary Table 6 — consequence sample by weather category | `M3_v4.ipynb` cell 0 (printed category table) |
| Section 4.2, Supplementary Figs. 3–5 — representative age/size, mutual adjustment, interaction tests | `03_fig3_accident_consequences/M3_reanalysis_complete.py` / `M3_v4.ipynb` cells 2–7; `fig_S3_S4_effects_at_values`, `fig_S5_joint_vs_separate` |
| Sections 5.1–5.3, Supplementary Fig. 6 — projections and quantile delta mapping | `00_data_preparation/04_climate_projections_and_bias_correction/wind_wave_QDM_test.ipynb` (regional), `global_qdm.py` (global); `fig_S6_qdm_evaluation` |
| Section 5.4, Supplementary Figs. 7–8 — regional threshold changes | `04_fig4_projected_occurrence/M4_test.ipynb`, `vis_pro_extreme_wind_wave.ipynb`; `fig_S7_S8_regional_threshold_changes` |
| Section 5.4, Supplementary Figs. 9–10 — seasonal changes for 2041–2060 and 2061–2080 | `04_fig4_projected_occurrence/M4_v1.ipynb` (B1/B2 tables); `fig_S9_S10_seasonal_heat` |
| Section 6.1, Supplementary Tables 7–8 — Arctic subregions, thickness interpolation check | `05_fig5_arctic/M5_pipeline.py` Cell 0; `05_fig5_arctic/check_sithick_interpolation.py` |
| Section 6.2, Supplementary Table 9 — POLARIS RIVs | `M5_pipeline.py` (`RIV_TABLE`, `riv`) |
| Section 6.3, Supplementary Figs. 11–12 — seasonal navigability, OW and PC7 | `M5_pipeline.py` Cell 8; `fig_S11_S12_navigability` |
| Section 6.4, Supplementary Figs. 13–14 — wave-data coverage, exposure decomposition | `M5_pipeline.py` Cells 9 and 11; `fig_S13_coverage`, `fig_S14_decomposition` |
| Section 6.5, Supplementary Table 10 — occurrence-index multipliers | `05_fig5_arctic/M5_vis_v2.ipynb` Cell A (`IRR_HS`, `IRR_U`) |

`SI_arctic_definitions.md` gives the exact definitions behind Supplementary Figs. 11–14 and Table 10.
