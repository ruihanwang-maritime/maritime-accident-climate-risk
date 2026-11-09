# -*- coding: utf-8 -*-
"""
make_map_days.py — builds the input table for Fig. 4a  _proj_global\tables\map_days.csv

Provenance
--------
Recovered from the 2026-09-03 analysis session log (originally the ad-hoc script mkfields2.py, not saved in the project folder).
Logic is identical to the version that produced map_days.csv (2026-09-03); only comments were added.

Inputs (all produced by global_qdm.py, in <DATA_ROOT>/new_climate/_proj_global)
  cells_sea.csv                 ocean 2° cell table: grid_id, clat, clon, flat_idx, good
  qdm_{kind}_{model}_{ssp}.nc   QDM bias-corrected daily max Hs / U10 (time, cell)
  cm_{kind}_{model}_{ssp}.nc    uncorrected model daily max, used to flag bias-correction artefacts

Output
  tables\map_days.csv  per cell:
    {kind}_{model}_{ssp}_{near|end}  2021–2040 / 2081–2100 mean annual exceedance days (Hs ≥ 6 m; U10 ≥ 17.2 m s⁻¹)
    spur_d   bias-correction artefact days: days exceeding after correction but with raw value below half the threshold (end of century, max over scenarios)
    masked   spur_d ≥ 1 day/yr and valid ocean cell and within 60S–60N → masked on the map
    in_domain  |lat| ≤ 60
M4_v2.ipynb Cell 0/1 reads this table, takes the two-model mean and draws the four Fig. 4a maps (end − near).
"""
import os
DATA_ROOT = os.environ.get('WRH_DATA_ROOT', 'G:/WRH_data')  # local data folder; see README
import os, numpy as np, pandas as pd, xarray as xr
np.seterr(all='ignore')

O = rf'{DATA_ROOT}/new_climate/_proj_global'
CS = pd.read_csv(os.path.join(O, 'cells_sea.csv')); GOOD = CS['good'].values.astype(bool)
LAT = CS['clat'].values

def da(p):
    d = xr.open_dataset(os.path.join(O, p)); return d[list(d.data_vars)[0]]

SPEC = {'wave': 6.0, 'wind': 17.2}                      # thresholds used in Fig. 4a
out = CS[['grid_id', 'clat', 'clon', 'flat_idx']].copy()
out['good'] = GOOD; out['in_domain'] = (LAT >= -60) & (LAT <= 60)
SPUR = np.zeros(len(CS))
for K, C in SPEC.items():
    for M in ('ACCESS', 'ECEARTH'):
        for S in ('ssp126', 'ssp585'):
            q = da(f'qdm_{K}_{M}_{S}.nc'); t = pd.to_datetime(q.time.values); v = q.values
            for lab, (y0, y1) in [('near', (2021, 2040)), ('end', (2081, 2100))]:
                k = (t.year >= y0) & (t.year <= y1)
                a = np.full(len(CS), np.nan); a[GOOD] = np.nansum(v[k] >= C, 0)[GOOD] / (y1 - y0 + 1)
                out[f'{K}_{M}_{S}_{lab}'] = a.round(3)
            ke = (t.year >= 2081) & (t.year <= 2100)
            R = da(f'cm_{K}_{M}_{S}.nc').values[ke]
            spur = (v[ke] >= C) & (R < C / 2)
            SPUR = np.maximum(SPUR, np.nansum(spur, 0) / 20.)
            del v, R, spur
            print(f'  {K} {M} {S} done', flush=True)
out['spur_d'] = SPUR.round(3)
out['masked'] = (SPUR >= 1.0) & out['good'].values & out['in_domain'].values
f = os.path.join(O, 'tables', 'map_days.csv'); out.to_csv(f, index=False)
print(f'\n-> {f}  {out.shape}   masked {int(out.masked.sum())} cells (union of scenarios)')

# area-weighted mean change in exceedance days within the domain (printed for checking)
D = out['good'].values & out['in_domain'].values & ~out['masked'].values
W = np.cos(np.radians(LAT))
for K, C in SPEC.items():
    for S in ('ssp126', 'ssp585'):
        e = (out[f'{K}_ACCESS_{S}_end'].values + out[f'{K}_ECEARTH_{S}_end'].values) / 2
        n = (out[f'{K}_ACCESS_{S}_near'].values + out[f'{K}_ECEARTH_{S}_near'].values) / 2
        print(f'  {K}>={C} {S}: near {np.nansum(n[D]*W[D])/W[D].sum():6.2f} -> end {np.nansum(e[D]*W[D])/W[D].sum():6.2f} d/yr '
              f'({(np.nansum(e[D]*W[D])/np.nansum(n[D]*W[D])-1)*100:+5.1f}%)')
