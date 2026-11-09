# Arctic Supplementary Figures 11–14 and Table 10: data definitions

All values come from the cached M5 outputs (`<DATA_ROOT>/new_climate/_M5/`) produced by `05_fig5_arctic/M5_pipeline.py`
(one run per climate model). Figures are drawn by `si_figures.py`; equation numbers below refer to the Supplementary
Information unless marked "main text".

Common grid and weights. Target grid = WW3 0.5° ocean cells north of 60°N (27,375 cells, ~350 spurious channel cells
removed). Cell area a_i = 3090 × cos(lat) km². Subregions R1–R4 (Supplementary Table 7) are the four longitude sectors
north of 66.5°N defined in `M5_pipeline.py` Cell 0.

## Supplementary Figs. 11–12 — monthly navigable area share (OW and PC7)
Pipeline Cell 8, `monthly_frac` / `period_month_table`; `fig_S11_S12_navigability`.
- Cell-day navigability: POLARIS RIO ≥ 0 for the vessel class (Supplementary Eq. 8; main-text Eq. 7), daily siconc and sithick.
- For each calendar month of each year: share = 100 × Σ_i a_i · (navigable days_i / days in month) / Σ_i a_i, over all ocean
  cells of the subregion (Supplementary Eq. 9).
- Period value = mean of the monthly share over the years of the period (1985–2014 from the historical run; future periods
  from the scenario run). Fig. 11 = SSP1–2.6, Fig. 12 = SSP5–8.5.

## Supplementary Fig. 13 — share of navigable days with valid wave data
Pipeline Cell 9, `coverage_table`; `fig_S13_coverage` (SSP5–8.5).
- Numerator: Σ_i a_i · (OW-navigable days with a finite WW3 Hs value in the month), pooled over all years of the period.
- Denominator: Σ_i a_i · (OW-navigable days in the month), pooled over the same years (Supplementary Eq. 10).
- Not reported when the navigable area in that month is below 5% of the subregion area (`MIN_NAV_FRAC = 0.05`).

## Supplementary Fig. 14 — decomposition of the change in severe-weather days
Pipeline Cell 11, `decomp` / `_agg`; `fig_S14_decomposition`. Area-weighted means over the newly navigable cells of each
subregion, in days per year:
- N = days with valid WW3 wave data; E = such days with Hs ≥ 2.5 m (or U10 ≥ 13.9 m/s); p = E / N.
- Window = ΔN · p_ref; weather frequency ("sea-state" in the code) = N_ref · Δp; interaction = ΔN · Δp; total = ΔE
  (Supplementary Eq. 11). Reference period 2041–2060, future period 2081–2100, same scenario.
- Closure check: window + frequency + interaction − total = 0 (asserted in the pipeline).

## Main-text Fig. 5c and Supplementary Table 10 — weather-associated occurrence index
`M5_vis_v2.ipynb` Cells A and D (`IRR_HS`, `IRR_U`).
- All five non-reference coefficients from the historical occurrence models are used, including the separate ≥9 m wave
  class: waves 1.0036, 1.0227, 1.0462, 1.0523, 1.1989; winds 1.0027, 1.0100, 1.0303, 1.0492, 1.1218 (Supplementary Table 10).
- Monthly multiplier M_it = exp(Σ_k ln(IRR_k) · n_itk), with n_itk = days with valid wave data in class k
  (main-text Eq. 10; Supplementary Eq. 12).
- Annual index = Σ a_i N_it M_it / Σ a_i N_it over the cells of the navigability group and the months of the year
  (main-text Eq. 11; Supplementary Eq. 13); NaN when the group's area-weighted mean valid wave days per year < 10.
- Plotted as a centred 10-year running mean requiring at least 5 valid annual values.

## Valid-day weights for wind and wave indices
Both indices use the same weights: nvalid = cell-days with a finite WW3 Hs value (`Wv['nvalid']`). The WW3 wind fields share
the wave model's ice mask; `M5_pipeline.py` Cell 5 asserts `nvalid_w == nvalid` for every model and scenario. WW3 does not
simulate waves in ice-covered cells; Cell 6 asserts that wave-valid cell-days exceeding POLARIS OW-navigable days are below
5% of wave-valid cell-days at the monthly scale.

## Sea-ice thickness interpolation check (Supplementary Table 8)
EC-Earth3 uses monthly sithick interpolated to daily. Validated on ACCESS-CM2 1985–2014 by recomputing POLARIS with daily vs
interpolated thickness (`05_fig5_arctic/check_sithick_interpolation.py`): OW +4.76% navigable cell-days, agreement 98.79%;
PC7 +2.02%, 98.10%.
